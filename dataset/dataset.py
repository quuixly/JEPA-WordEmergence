from torch.utils.data import Dataset
import os
import pickle
from tqdm import tqdm
import torch
from game.othello import GameBoard, Piece


class OthelloDataset(Dataset):
    def __init__(self, train=True, num_samples=-1):
        self.bos_token_id = 0
        self.pad_token_id = 61
        self.eos_token_id = 62
        self.max_length = 63

        data_path = "dataset/train/" if train else "dataset/test/"

        files = sorted([
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".pkl")
        ])

        forbidden = {27, 28, 35, 36}
        position_to_id = {}
        current_id = 1
        for r in range(8):
            for c in range(8):
                pos_idx = r * 8 + c
                if pos_idx not in forbidden:
                    pos_name = f"{chr(c + ord('A'))}{r + 1}"
                    position_to_id[pos_name] = current_id
                    current_id += 1

        all_inputs = []
        all_targets = []

        print("Preprocessing dataset into tensors...")

        for file in tqdm(files, desc="Loading & preprocessing"):
            with open(file, "rb") as f:
                batches = pickle.load(f)

            for sequence in batches:
                moves = [
                    position_to_id[pos]
                    for _, pos in sequence
                    if pos in position_to_id
                ]

                board = GameBoard()

                input_ids = [self.bos_token_id] + moves[:-1]
                target_ids = moves[:]

                current_player = Piece.BLACK
                for _, pos in sequence:
                    board.add_piece(current_player, pos)
                    current_player = (
                        Piece.WHITE
                        if current_player == Piece.BLACK
                        else Piece.BLACK
                    )

                has_moves_current = len(board.get_legal_moves(current_player)) > 0
                opponent = (
                    Piece.BLACK if current_player == Piece.WHITE else Piece.WHITE
                )
                has_moves_opp = len(board.get_legal_moves(opponent)) > 0

                if not has_moves_current and not has_moves_opp:
                    if len(target_ids) < self.max_length:
                        target_ids.append(self.eos_token_id)
                    else:
                        target_ids[-1] = self.eos_token_id

                # padding
                input_ids += [self.pad_token_id] * (
                    self.max_length - len(input_ids)
                )
                target_ids += [self.pad_token_id] * (
                    self.max_length - len(target_ids)
                )

                all_inputs.append(input_ids[:self.max_length])
                all_targets.append(target_ids[:self.max_length])

                if num_samples > 0 and len(all_inputs) >= num_samples:
                    break

            if num_samples > 0 and len(all_inputs) >= num_samples:
                break

        self.inputs = torch.tensor(all_inputs, dtype=torch.long)
        self.targets = torch.tensor(all_targets, dtype=torch.long)

        cache_path = "dataset_cache.pt"
        if os.path.exists(cache_path):
            data = torch.load(cache_path)
            self.inputs, self.targets = data['in'], data['out']
        else:
            torch.save({'in': self.inputs, 'out': self.targets}, cache_path)

        self.inputs.share_memory_()
        self.targets.share_memory_()

        print("Dataset ready.")
        print("Shape:", self.inputs.shape)

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]