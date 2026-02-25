from torch.utils.data import Dataset
import torch
import os

class OthelloDataset(Dataset):
    def __init__(self, train=True):
        cache_path = "./dataset/dataset_cache(1).pt" if train else "./dataset/dataset_cache_test.pt"

        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        data = torch.load(cache_path)
        self.inputs, self.targets = data['in'].share_memory_(), data['out'].share_memory_()

        print(f"Loaded dataset from cache: {cache_path}")
        print("Shape:", self.inputs.shape)

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, index):
        return self.inputs[index].long(), self.targets[index].long()