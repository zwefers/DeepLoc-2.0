import os
import tqdm
import pandas as pd
import numpy as np
import pickle
import torch
from src.utils import ModelAttributes
from src.data import DataloaderHandler
from src.metrics import *

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "cpu"
    dtype=torch.bfloat16
else:
    device = "cpu"
    dtype=torch.bfloat16

def predict_sl_values(dataloader, model):
    output_dict = {}
    annot_dict = {}
    pool_dict = {}
    with torch.no_grad():
      for i, (toks, lengths, np_mask, targets, targets_seq, labels) in tqdm.tqdm(enumerate(dataloader)):
        with torch.autocast(device_type=device,dtype=dtype):
            y_pred, y_pool, y_attn = model.predict(toks.to(device), lengths.to(device), np_mask.to(device))
        x = torch.sigmoid(y_pred).float().cpu().numpy()
        for j in range(len(labels)):
            if len(labels) == 1:
                output_dict[labels[j]] = x
                pool_dict[labels[j]] = y_pool.float().cpu().numpy()
                annot_dict[labels[j]] = y_attn[:lengths[j]].float().cpu().numpy()
            else:
                output_dict[labels[j]] = x[j]
                pool_dict[labels[j]] = y_pool[j].float().cpu().numpy()
                annot_dict[labels[j]] = y_attn[j,:lengths[j]].float().cpu().numpy()

    output_df = pd.DataFrame(output_dict.items(), columns=['ACC', 'preds'])
    annot_df = pd.DataFrame(annot_dict.items(), columns=['ACC', 'pred_annot'])
    pool_df = pd.DataFrame(pool_dict.items(), columns=['ACC', 'embeds'])
    return output_df.merge(annot_df).merge(pool_df)
    
def generate_sl_outputs(
        modelname,
        model_attrs: ModelAttributes, 
        datahandler: DataloaderHandler,
        thresh_type="mcc", 
        inner_i="1Layer", 
        reuse=False, 
        test=False):
    
    num_classes=datahandler.num_classes
    threshold_dict = {}
        
    for i in range(5):
        print("Generating output for ensemble model", i)
        if test:
            j=0 #HOU_testset only has 1 fold
        else:
            j=i
        model_path = f"{model_attrs.save_path}/{i}_{modelname}.ckpt"
        print(f"loaded model: {model_path}")
        model = model_attrs.class_type.load_from_checkpoint(model_path, num_classes=model_attrs.num_classes,
                                                                pos_weights=model_attrs.pos_weights).to(device).eval()
        pred_savename = modelname
        if test: 
            pred_savename = f"{pred_savename}_hou"

        if not test:
            dataloader, data_df = datahandler.get_partition_dataloader_inner(i)
            df_name = os.path.join(model_attrs.outputs_save_path, f"{i}_{pred_savename}_trainout.pkl")
            if not os.path.exists(df_name):
                pred_df = predict_sl_values(dataloader, model)
                print(f"Saving output to: {df_name}")
                pred_df.to_pickle(df_name)
            else:
                print(f"Loading outputs: {df_name}")
                pred_df = pd.read_pickle(os.path.join(df_name))

            if thresh_type == "roc":
                thresholds = get_optimal_threshold(pred_df, data_df, num_classes)
            elif thresh_type == "pr":
                thresholds = get_optimal_threshold_pr(pred_df, data_df, num_classes)
            else:
                thresholds = get_optimal_threshold_mcc(pred_df, data_df, num_classes)
            threshold_dict[f"{i}_{pred_savename}"] = thresholds
    
        if not os.path.exists(os.path.join(model_attrs.outputs_save_path, f"{i}_{pred_savename}_testout.pkl")):
            dataloader, data_df = datahandler.get_partition_dataloader(j)
            output_df = predict_sl_values(dataloader, model)
            output_df.to_pickle(os.path.join(model_attrs.outputs_save_path, f"{i}_{pred_savename}_testout.pkl"))

    if not test:
        with open(os.path.join(model_attrs.outputs_save_path, f"{modelname}_thresholds_sl_{thresh_type}.pkl"), "wb") as f:
            pickle.dump(threshold_dict, f)

def predict_ss_values(X, model):
    X_tensor = torch.tensor(X, device=device).float()
    y_preds = torch.sigmoid(model(X_tensor))
    return y_preds.detach().cpu().numpy()

def generate_ss_outputs(
        model_attrs: ModelAttributes, 
        datahandler: DataloaderHandler, 
        thresh_type="mcc", 
        inner_i="1Layer", 
        reuse=False):
    
    threshold_dict = {}
    if not os.path.exists(f"{model_attrs.outputs_save_path}"):
        os.makedirs(f"{model_attrs.outputs_save_path}")
    for outer_i in range(5):
        print("Generating output for ensemble model", outer_i)
        X_train, y_train, X_test, y_test = datahandler.get_swissprot_ss_xy(model_attrs.outputs_save_path, outer_i)
        path = f"{model_attrs.save_path}/signaltype/{outer_i}.ckpt"
        model = SignalTypeMLP.load_from_checkpoint(path).to(device).eval()
        
        y_train_preds = predict_ss_values(X_train, model)
        thresh = np.zeros((9,))
        threshold_dict = {}
        #print("thresholds")
        for type_i in range(9):
            thresh[type_i] = get_best_threshold_mcc(y_train[:, type_i], y_train_preds[:, type_i])
            threshold_dict[SS_CATEGORIES[type_i+1]] = thresh[type_i]
            #print(SS_CATEGORIES[type_i+1], thresh[type_i])
        y_test_preds = predict_ss_values(X_test, model)
        pickle.dump(y_test_preds, open(f"{model_attrs.outputs_save_path}/ss_{outer_i}.pkl", "wb"))

    with open(os.path.join(model_attrs.outputs_save_path, f"thresholds_ss_mcc.pkl"), "wb") as f:
        pickle.dump(threshold_dict, f)
