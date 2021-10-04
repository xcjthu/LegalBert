from .model.CNN import TextCNN
from .model.Bert import Bert
from .VanillaLFM import VanillaLFM, VanillaBert
from .MemBERT.RecurrentTrans import RecurrentTransMLM

model_list = {
    "CNN": TextCNN,
    "BERT": Bert,
    "VanillaLFM": VanillaLFM,
    "VanillaBert": VanillaBert,
    "RecurrentMLM": RecurrentTransMLM
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
