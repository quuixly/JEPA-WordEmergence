from torch.utils.data import Dataset
import torch
import os

PADDING = -100
MAX_MOVES = 60

class OthelloDataset(Dataset):
    def __init__(self, train=True):
        cache_path = "./dataset/othello_dataset.pt" if train else "./dataset/dataset_cache_test.pt"

        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        data = torch.load(cache_path)
        num_games, max_len = data.shape

        self.inputs = data
        self.targets = torch.full_like(self.inputs, PADDING)
        self.targets[:, :-1] = self.inputs[:, 1:]
        print(self.inputs.min(), self.inputs.max())
        print(self.targets.min(), self.targets.max())
        self.inputs.share_memory_()
        self.targets.share_memory_()

        print(f"Loaded dataset from cache: {cache_path}")
        print("Number of games:", num_games)
        print("Sequence length (inputs & targets):", self.inputs.shape[1])

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        x = self.inputs[idx].long()
        y = self.targets[idx].long()
        return x, y