from .model.CNN import TextCNN
from .model.Bert import Bert
from .VanillaLFM import VanillaLFM, VanillaBert, VanillaDimRedBERT
from .DocLawLFM import DocLawLFM
from .ContrastiveLawLFM import ContrastiveLawLFM
from .TokenSkipLFMPretrain import TokenSkipLFMPretrain
from .UnContrastiveLFM import UnContrastiveLFM
from .UnContrastiveLFMVer2 import UnContrastiveLFMVer2
model_list = {
    "CNN": TextCNN,
    "BERT": Bert,
    "VanillaLFM": VanillaLFM,
    "VanillaBert": VanillaBert,
    "DocLaw": DocLawLFM,
    "ContrastiveLaw": ContrastiveLawLFM,
    "DimRedBERT": VanillaDimRedBERT,
    "TokenSkip": TokenSkipLFMPretrain,
    "UnContrastive": UnContrastiveLFM,
    "UnContrastiveVer2": UnContrastiveLFMVer2
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
