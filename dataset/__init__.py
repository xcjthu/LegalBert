from dataset.JsonFromFiles import JsonFromFilesDataset
from .IndexedDataset import make_dataset

dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "IndexedDataset": make_dataset,
}
