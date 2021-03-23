from dataset.JsonFromFiles import JsonFromFilesDataset
from .IndexedDataset import make_dataset
from .FullTokenDataset import FullTokenDataset,MultiDocDataset

dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "IndexedDataset": make_dataset,
    "FullTokenDataset": FullTokenDataset,
    "MultiDocDataset": MultiDocDataset,
}
