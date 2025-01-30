from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.data import *
from src.utils import *
from src.eval_utils import *
from src.embedding import *
from src.metrics import *
import argparse
import subprocess
import os
import warnings
import ast
import pandas as pd
import yaml

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def make_DL2_df(original_csv, level, categories):
    df = pd.read_csv(original_csv)
    if "fold" not in df.columns:
        df["fold"] = 0
    one_hot = []
    targets = []
    for locs in df[f"level{level}"].str.split(";").to_list():
        temp = [1 if loc in locs else 0 for loc in categories]
        one_hot.append([1 if loc in locs else 0 for loc in categories])
        targets.append(temp)
    one_hot = np.array(one_hot)
    DL2_df = pd.DataFrame(one_hot, columns=categories)
    if "sequence" in df.columns:
        DL2_df["Sequence"] = df["sequence"]
    elif "Sequence" in df.columns:
        DL2_df["Sequence"] = df["Sequence"]
    else: Exception("No seq col in dataframe")
    if "ensembl_ids" in df.columns:
        acc = "ensembl_ids"
    elif "id" in df.columns:
        acc = "id"
    elif "uniprot_id" in df.columns:
        acc = "uniprot_id"
    else: raise Exception("No id col in dataframe")
    DL2_df.insert(0,'ACC','')
    DL2_df["ACC"] = df[acc] 
    DL2_df["Partition"] = df["fold"] 
    DL2_df["Target"] = targets
    DL2_df["ANNOT"] = pd.NA
    DL2_df["Types"] = pd.NA
    DL2_df["TargetAnnot"] = 0
    return DL2_df



def train_model(modelname:str, model_attrs: ModelAttributes, datahandler:DataloaderHandler, outer_i: int, pos_weights: torch.tensor):
    train_dataloader, val_dataloader = datahandler.get_train_val_dataloaders(outer_i)
    num_classes = datahandler.num_classes #zoe

    checkpoint_callback = ModelCheckpoint(
        monitor='bce_loss',
        dirpath=model_attrs.save_path,
        filename=  f"{i}_{modelname}",
        save_top_k=1,
        every_n_epochs=1,
        save_last=False,
        save_weights_only=True
    )

    early_stopping_callback = EarlyStopping(
         monitor='bce_loss',
         patience=5, 
         mode='min'
    )

    # Initialize trainer
    trainer = pl.Trainer(max_epochs=14, 
                        default_root_dir=model_attrs.save_path + f"/{i}_{modelname}",
                        check_val_every_n_epoch = 1,
                        callbacks=[
                            checkpoint_callback, 
                            early_stopping_callback
                        ],
                        precision=16,
                        accelerator="auto")
    clf = model_attrs.class_type(model_attrs.num_classes, pos_weights=pos_weights) #zoe (get this from DataloaderHandler?)
    trainer.fit(clf, train_dataloader, val_dataloader)
    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m","--model", 
        default="Fast",
        choices=['Accurate', 'Fast', 'seq2loc'],
        type=str,
        help="Model to use."
    )

    parser.add_argument(
        "-l","--level", 
        default=0,
        choices=[0, 1, 2, 3], #0 is for original DeepLoc2
        type=int,
        help="Level of localization categories"
    )

    parser.add_argument(
        "-d","--dataset",
        default="",
        type=str,
        help="training data csv"
    )

    parser.add_argument(
        "-t","--test_dataset",
        default="",
        type=str,
        help="test data csv"
    )

    args = parser.parse_args()

    

    if len(args.dataset) == 0:
        data_df=None
        data_code = "orig"
    else:
        CATEGORIES_YAML = load_config("data_files/seq2loc/level_classes.yaml")
        categories = CATEGORIES_YAML[f"level{args.level}"]
        data_df = make_DL2_df(args.dataset, args.level, categories)
        data_codes = {"hpa_trainset.csv": "hpa",
                  "uniprot_trainset.csv": "uniprot",
                  "hpa_uniprot_combined_human_trainset.csv": "combined_human",
                  "hpa_uniprot_combined_trainset.csv": "combined"}
        data_code = data_codes[args.dataset.split("/")[-1]]
        
    

    level_numclasses = {0:11, 1:21, 2:10, 3:8}
    num_classes = level_numclasses[args.level]
    if len(args.dataset) == 0:
        data_df=None


    #TODO: Let the clip_len be variable
    CLIP_LEN=4000
    def clip(seq, clip_len):
        assert clip_len % 2 == 0
        if len(seq) > clip_len:
            seq = seq[ : clip_len//2] + seq[-clip_len//2 : ]
        return seq
    #CLIP sequences in metadata
    if data_df is not None:
        data_df.Sequence = data_df.Sequence.apply(lambda seq: clip(seq, CLIP_LEN))
    
    
    #SET pos_weights
    if args.level==0: # pos_weights defined by deeploc
        pos_weights = None
    else:
        pos_weights = 1/(torch.tensor(data_df.Target.to_list(), dtype=torch.float32).mean(axis=0)+ 1e-5)

    model_attrs = get_train_model_attributes(model_type=args.model, num_classes=num_classes, pos_weights=pos_weights)
    if not os.path.exists(model_attrs.embedding_file):
        print("Embeddings not found, generating......")
        generate_embeddings(model_attrs)
        print("Embeddings created!")
    else:
        print("Using existing embeddings")
    
    if not os.path.exists(model_attrs.embedding_file):
        raise Exception("Embeddings could not be created. Verify that data_files/embeddings/<MODEL_DATASET> is deleted")
    
    datahandler = DataloaderHandler(
        clip_len=model_attrs.clip_len, 
        alphabet=model_attrs.alphabet, 
        embedding_file=model_attrs.embedding_file,
        embed_len=model_attrs.embed_len,
        num_classes=num_classes,
        metadata=data_df #zoe
    )

    print("Training subcellular localization models")
    modelname = f"1Layer_{data_code}_level{args.level}"
    for i in range(0, 5):
        print(f"Training model {i+1} / 5")
        if not os.path.exists(os.path.join(model_attrs.save_path, f"{i}_{modelname}.ckpt")):
            train_model(modelname, model_attrs, datahandler, i, pos_weights)
    print("Finished training subcellular localization models")

    print("Using trained models to generate outputs for signal prediction training")
    generate_sl_outputs(modelname, model_attrs=model_attrs, datahandler=datahandler)
    print("Generated outputs! Can train sorting signal prediction now")
    
    print("Computing subcellular localization performance on swissprot CV dataset")
    calculate_sl_metrics(modelname, model_attrs, datahandler=datahandler)

    if len(args.test_dataset) > 0:
        print(f"Testing {args.test_dataset}")
        test_df = make_DL2_df(args.test_dataset, args.level, categories)
        test_df.Sequence = test_df.Sequence.apply(lambda seq: clip(seq, CLIP_LEN))
        test_datahandler = DataloaderHandler(
            clip_len=model_attrs.clip_len, 
            alphabet=model_attrs.alphabet, 
            embedding_file=model_attrs.embedding_file,
            embed_len=model_attrs.embed_len,
            num_classes=num_classes,
            metadata=test_df #zoe
        )
        generate_sl_outputs(modelname, model_attrs=model_attrs, datahandler=datahandler, test=True)
        calculate_sl_metrics(modelname, model_attrs, datahandler=datahandler, test=True)
