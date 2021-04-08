from dataset.JsonFromFiles import JsonFromFilesDataset
from .IndexedDataset import make_dataset
from .FullTokenDataset import FullTokenDataset,MultiDocDataset
from .DocLawDataset import DocLawDataset
from .DocLawDataset import NSPDocLawDataset
from .ContrastiveLawDataset import ContrastiveLawDataset

dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "IndexedDataset": make_dataset,
    "FullTokenDataset": FullTokenDataset,
    "MultiDocDataset": MultiDocDataset,
    "DocLaw": DocLawDataset,
    "NSPLaw": NSPDocLawDataset,
    "ContrastiveLaw": ContrastiveLawDataset,
}
