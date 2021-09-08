from .model.CNN import TextCNN
from .model.Bert import Bert
from .VanillaLFM import VanillaLFM, VanillaBert, VanillaDimRedBERT
from .DocLawLFM import DocLawLFM
from .ContrastiveLawLFM import ContrastiveLawLFM
from .TokenSkipLFMPretrain import TokenSkipLFMPretrain

model_list = {
    "CNN": TextCNN,
    "BERT": Bert,
    "VanillaLFM": VanillaLFM,
    "VanillaBert": VanillaBert,
    "DocLaw": DocLawLFM,
    "ContrastiveLaw": ContrastiveLawLFM,
    "DimRedBERT": VanillaDimRedBERT,
    "TokenSkip": TokenSkipLFMPretrain
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
