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
from itertools import groupby

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_signal_locs(s):
    signals = []
    start=0
    for i, group in groupby(s):
        len_signal = len(list(group))
        if i != 0:
            idx = [i for i in range(start, start+len_signal)]
            signals.append((i, idx))
        start += len_signal
    return signals

def signal_loc_to_annot(signal_locs, seq_len):
    s = [0 for i in range(seq_len)]
    for i, idxs in signal_locs:
        for j in idxs:
            s[j] = i
    return s

def add_sorting_signals(sorting_df, trainset):
    # Add sorting signals from DL2 original data to our data
    # only 868/1874 sorting signals are kept for uniprot_trainset
    sorting_df = sorting_df.rename({"Sequence": "sequence"}, axis=1)

    temp = sorting_df.merge(
        trainset, how="outer", left_on="ACC", right_on="uniprot_id", 
        suffixes=["_dl2", "_uni"])
    temp = temp[temp.level1.notna()].reset_index(drop=True)

    #Some sequences do not agree, so can't directly take sorting signals
    disagreement = temp[
        (temp.sequence_dl2 != temp.sequence_uni) & (temp.ANNOT.notna())
        ]
    for i, row in disagreement.iterrows():
        annot = row.ANNOT
        if isinstance(annot, np.ndarray):
            their_seq = row.sequence_dl2
            our_seq = row.sequence_uni
            signal_locs = get_signal_locs(annot)
            our_signal_locs = []

            #Check if sorting signal is in our sequence at different position
            for j, idxs in signal_locs:
                signal_seq = "".join([their_seq[j] for j in idxs])
                our_idx = our_seq.find(signal_seq)

                #If it is, then get new annotation
                if our_idx != -1:
                    our_idxs = [
                        k for k in range(our_idx, our_idx+len(signal_seq))
                        ]
                    our_signal_locs.append((j, our_idxs))
            if our_signal_locs: #if not empty
                s = signal_loc_to_annot(signal_locs, len(our_seq))
                s = np.array(s)
                temp.at[i, "ANNOT"] = s

    #temp = temp.fillna(0)
    temp = temp.drop(["ACC", "sequence_dl2"], axis=1)
    temp = temp.rename({"sequence_uni": "Sequence"}, axis=1)
    temp = temp[["ANNOT", "uniprot_id", "Sequence", "level1", "level2", "level3", "fold"]]
    return temp

def make_DL2_df(df, level, categories, clip_len):
    def clip_middle_np(x):
        if isinstance(x, np.ndarray) and len(x)>clip_len:
            x = np.concatenate((x[:clip_len//2],x[-clip_len//2:]), axis=0)
        return x
    def clip_middle(x):
        if type(x)==str  and len(x)>clip_len:
            x = x[:clip_len//2] + x[-clip_len//2:]
        return x
    #def s_to_np(s):
        #if type(s) == str:
            #s = np.array([float(i) for i in s], dtype=np.float64)
        #return s
    

    #Get target locations
    one_hot = []
    targets = []
    for locs in df[f"level{level}"].str.split(";").to_list():
        temp = [1 if loc in locs else 0 for loc in categories]
        one_hot.append([1 if loc in locs else 0 for loc in categories])
        targets.append(temp)
    one_hot = np.array(one_hot)
    DL2_df = pd.DataFrame(one_hot, columns=categories)
    
    #Sequence column
    if "sequence" in df.columns:
        DL2_df["Sequence"] = df["sequence"]
    elif "Sequence" in df.columns:
        DL2_df["Sequence"] = df["Sequence"]
    else: Exception("No seq col in dataframe")
    DL2_df["Sequence"] = DL2_df["Sequence"].apply(clip_middle)
    
    #ID column
    if "uniprot_id" in df.columns:
        acc = "uniprot_id"
    else: raise Exception("No id col in dataframe")
    DL2_df.insert(0,'ACC','')
    DL2_df["ACC"] = df[acc] 
 
    if "fold" not in df.columns:
        df["fold"] = 0
    DL2_df["Partition"] = df["fold"]

    #Sorting Signals
    if "ANNOT" in df.columns and "Types" in df.columns:
        DL2_df["Target"] = targets
        DL2_df["ANNOT"] = df["ANNOT"]
        DL2_df["Types"] = df["Types"]
        DL2_df["TargetAnnot"] = DL2_df["ANNOT"].apply(clip_middle_np)
    else:
         DL2_df["Target"] = targets
         DL2_df["ANNOT"] = [
            np.zeros(len(s), dtype=np.float64) for s in DL2_df["Sequence"]
            ]
         DL2_df["TargetAnnot"] = DL2_df["ANNOT"]
    
    return DL2_df


def train_model(
    modelname:str, model_attrs: ModelAttributes, datahandler:DataloaderHandler, 
    outer_i: int, pos_weights: torch.tensor):
    train_dataloader, val_dataloader = datahandler.get_train_val_dataloaders(outer_i)
    num_classes = datahandler.num_classes

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
    clf = model_attrs.class_type(model_attrs.num_classes, pos_weights=pos_weights)
    trainer.fit(clf, train_dataloader, val_dataloader)
    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m","--model", 
        default="Fast",
        choices=['Accurate', 'Fast', 'seq2loc-prott5', 'seq2loc-ems1'],
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

    parser.add_argument(
        "-c","--clip_len",
        default=4000,
        type=int,
        help="sequence length to use"
    )
    
    parser.add_argument(
        "-y","--classes_yaml",
        default="/hai/scratch/zwefers/seq2loc/deeploc2/data_files/seq2loc/level_classes.yaml",
        type=str,
        help="file to define categories for each level"
    )

    args = parser.parse_args()

    

    if len(args.dataset) == 0:
        trainset=None
        data_code = "orig"
        categories=CATEGORIES
    else:
        CATEGORIES_YAML = load_config(args.classes_yaml)
        categories = CATEGORIES_YAML[f"level{args.level}"]

        trainset = pd.read_csv(args.dataset)

        sorting_signals = pd.read_pickle("data_files/multisub_ninesignals.pkl")
        trainset = add_sorting_signals(sorting_signals, trainset)
        trainset = make_DL2_df(trainset, args.level, categories, args.clip_len)

        data_codes = {
            "hpa_trainset.csv": "hpa",
            "uniprot_trainset.csv": "uniprot",
            "uniprot_trainset_wsortsigs.csv": "uni_sortsig",
            "hpa_uniprot_combined_human_trainset.csv": "combined_human",
            "hpa_uniprot_combined_trainset.csv": "combined"
            }
        data_code = data_codes[args.dataset.split("/")[-1]]


    level_numclasses = {0:11, 1:21, 2:10, 3:8}
    num_classes = level_numclasses[args.level]


    def clip(seq, clip_len):
        assert clip_len % 2 == 0
        if len(seq) > clip_len:
            seq = seq[ : clip_len//2] + seq[-clip_len//2 : ]
        return seq
    
    #Set pos_weights
    if args.level==0: # pos_weights defined by deeploc
        pos_weights = None
    else:
        pos_weights = 1/(
            torch.tensor(
                trainset.Target.to_list(), dtype=torch.float32
            ).mean(axis=0)+ 1e-5)

    model_attrs = get_train_model_attributes(
        model_type=args.model, num_classes=num_classes, pos_weights=pos_weights)
    if not os.path.exists(model_attrs.embedding_file):
        print("Embeddings not found, generating......")
        generate_embeddings(model_attrs)
        print("Embeddings created!")
    else:
        print("Using existing embeddings")
    
    if not os.path.exists(model_attrs.embedding_file):
        raise Exception(
            "Embeddings could not be created. Verify that data_files/embeddings/<MODEL_DATASET> is deleted"
            )
    
    datahandler = DataloaderHandler(
        clip_len=model_attrs.clip_len, 
        alphabet=model_attrs.alphabet, 
        embedding_file=model_attrs.embedding_file,
        embed_len=model_attrs.embed_len,
        num_classes=num_classes,
        metadata=trainset #zoe
    )

    print("Training subcellular localization models")
    modelname = f"1Layer_{data_code}_level{args.level}"
    for i in range(0, 5):
        print(f"Training model {i+1} / 5")
        if not os.path.exists(os.path.join(model_attrs.save_path, f"{i}_{modelname}.ckpt")):
            train_model(modelname, model_attrs, datahandler, i, pos_weights)
        else:
            print("Model already trained")
    print("Finished training subcellular localization models")

    print("Using trained models to generate outputs")
    generate_sl_outputs(
        modelname, model_attrs=model_attrs, datahandler=datahandler
        )

    if len(args.test_dataset) > 0:
        print(f"Testing on {args.test_dataset}")
        test_df = pd.read_csv(args.test_dataset)
        test_df = make_DL2_df(test_df, args.level, categories, args.clip_len)
        test_datahandler = DataloaderHandler(
            clip_len=model_attrs.clip_len, 
            alphabet=model_attrs.alphabet, 
            embedding_file=model_attrs.embedding_file,
            embed_len=model_attrs.embed_len,
            num_classes=num_classes,
            metadata=test_df #zoe
        )
        generate_sl_outputs(
            modelname, model_attrs=model_attrs, datahandler=test_datahandler, 
            test=True
            )