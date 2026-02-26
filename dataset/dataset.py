from torch.utils.data import Dataset
import torch
import os

class OthelloDataset(Dataset):
    def __init__(self, train=True):
        cache_path = "./dataset/dataset_cache(1).pt" if train else "./dataset/dataset_cache_test.pt"

        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        data = torch.load(cache_path)
        inputs, targets = data['in'], data['out']

        print(f"Loaded dataset from cache: {cache_path}")
        print("Original shape:", inputs.shape)


        inputs_flat = ["_".join(map(str, seq.tolist())) for seq in inputs]
        unique_map = {}
        unique_indices = []

        for i, s in enumerate(inputs_flat):
            if s not in unique_map:
                unique_map[s] = True
                unique_indices.append(i)

        self.inputs = inputs[unique_indices].share_memory_()
        self.targets = targets[unique_indices].share_memory_()

        print("Shape after removing duplicates:", self.inputs.shape)
        print("Removed duplicates:", inputs.shape[0] - self.inputs.shape[0])
        print("Percent duplicates removed: {:.2f}%".format(
            100 * (inputs.shape[0] - self.inputs.shape[0]) / inputs.shape[0]
        ))

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, index):
        return self.inputs[index].long(), self.targets[index].long()