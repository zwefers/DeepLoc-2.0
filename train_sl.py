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

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

def train_model(model_attrs: ModelAttributes, datahandler:DataloaderHandler, outer_i: int, pos_weights: torch.tensor):
    train_dataloader, val_dataloader = datahandler.get_train_val_dataloaders(outer_i)
    num_classes = datahandler.num_classes #zoe

    checkpoint_callback = ModelCheckpoint(
        monitor='bce_loss',
        dirpath=model_attrs.save_path,
        filename= f"{outer_i}_1Layer",
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
                        default_root_dir=model_attrs.save_path + f"/{outer_i}_1Layer",
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
        "-d","--data",
        default="",
        type=str,
        help="training data csv"
    )

    args = parser.parse_args()

    level_numclasses = {0:11, 1:21, 2:10, 3:8}
    num_classes = level_numclasses[args.level]
    if len(args.data) != 0:
        data_df = pd.read_csv(args.data)
        data_df.Target = data_df.Target.apply(ast.literal_eval)
    else:
        data_df=None


    model_attrs = get_train_model_attributes(model_type=args.model, num_classes=num_classes)
    if not os.path.exists(model_attrs.embedding_file):
        print("Embeddings not found, generating......")
        generate_embeddings(model_attrs)
        print("Embeddings created!")
    else:
        print("Using existing embeddings")
    
    if not os.path.exists(model_attrs.embedding_file):
        raise Exception("Embeddings could not be created. Verify that data_files/embeddings/<MODEL_DATASET> is deleted")


    #CLIP sequences in metadata
    def clip(seq, clip_len):
        assert clip_len % 2 == 0
        if len(seq) > clip_len:
            seq = seq[ : clip_len//2] + seq[-clip_len//2 : ]
        return seq
    if data_df is not None:
        data_df.Sequence = data_df.Sequence.apply(lambda seq: clip(seq, model_attrs.clip_len))

    #SET pos_weights
    if args.level==0: # pos_weights defined by deeploc
        pos_weights = torch.tensor([1,1,1,3,2.3,4,9.5,4.5,6.6,7.7,32])
    else:
        pos_weights = 1/(torch.tensor(data_df.Target.to_list(), dtype=torch.float32).mean(axis=0)+ 1e-5)
    

    datahandler = DataloaderHandler(
        clip_len=model_attrs.clip_len, 
        alphabet=model_attrs.alphabet, 
        embedding_file=model_attrs.embedding_file,
        embed_len=model_attrs.embed_len,
        num_classes=num_classes,
        metadata=data_df #zoe
    )


    print("Training subcellular localization models")
    for i in range(0, 5):
        print(f"Training model {i+1} / 5")
        if not os.path.exists(os.path.join(model_attrs.save_path, f"{i}_1Layer.ckpt")):
            train_model(model_attrs, datahandler, i, pos_weights)
    print("Finished training subcellular localization models")

    print("Using trained models to generate outputs for signal prediction training")
    generate_sl_outputs(model_attrs=model_attrs, datahandler=datahandler)
    print("Generated outputs! Can train sorting signal prediction now")


    print("Computing subcellular localization performance on swissprot CV dataset")
    calculate_sl_metrics(model_attrs=model_attrs, datahandler=datahandler)
