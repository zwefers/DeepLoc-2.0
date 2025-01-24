from src.model import *
from src.data import DataloaderHandler
import pickle
from transformers import T5EncoderModel, T5Tokenizer, logging
import os
import torch
class ModelAttributes:
    def __init__(self, 
                 model_type: str,
                 class_type: pl.LightningModule, 
                 alphabet, 
                 embedding_file: str, 
                 save_path: str,
                 outputs_save_path: str,
                 clip_len: int,
                 embed_len: int,
                 num_classes: int,
                 pos_weights=None) -> None:
        self.model_type = model_type
        self.class_type = class_type 
        self.alphabet = alphabet
        self.embedding_file = embedding_file
        self.save_path = save_path
        if not os.path.exists(f"{self.save_path}"):
            os.makedirs(f"{self.save_path}")
        self.ss_save_path = os.path.join(self.save_path, "signaltype")
        if not os.path.exists(f"{self.ss_save_path}"):
            os.makedirs(f"{self.ss_save_path}")

        self.outputs_save_path = outputs_save_path

        if not os.path.exists(f"{outputs_save_path}"):
            os.makedirs(f"{outputs_save_path}")
        self.clip_len = clip_len
        self.embed_len = embed_len
        self.num_classes = num_classes
        self.pos_weights=pos_weights
        

def get_train_model_attributes(model_type, num_classes, pos_weights=None):
    if model_type == FAST:
        with open("models/ESM1b_alphabet.pkl", "rb") as f:
            alphabet = pickle.load(f)
        return ModelAttributes(
            model_type,
            ESM1bFrozen,
            alphabet,
            EMBEDDINGS[FAST]["embeds"],
            "models/models_esm1b",
            "outputs/esm1b/",
            1022,
            1280,
            num_classes,
            torch.tensor([1,1,1,3,2.3,4,9.5,4.5,6.6,7.7,32])
        )
    elif model_type == ACCURATE:
        alphabet = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
        
        return ModelAttributes(
            model_type,
            ProtT5Frozen,
            alphabet,
            EMBEDDINGS[ACCURATE]["embeds"],            
            "models/models_prott5",
            "outputs/prott5/",
            4000,
            1024,
            num_classes,
            torch.tensor([1,1,1,3,2.3,4,9.5,4.5,6.6,7.7,32])
        )
    elif model_type == SEQ2LOC:
        alphabet = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
        
        return ModelAttributes(
            model_type,
            ProtT5Frozen,
            alphabet,
            EMBEDDINGS[SEQ2LOC]["embeds"],            
            "models/seq2locbench",
            "outputs/seq2locbench/",
            4000,
            1024,
            num_classes,
            pos_weights
        )
    else:
        raise Exception("wrong model type provided expected Fast,Accurate got", model_type)
    

