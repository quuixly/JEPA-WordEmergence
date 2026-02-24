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
        self.train = train
        self.num_samples = num_samples
        self.data_path = "dataset/train/" if train else "dataset/test/"

        self.files = sorted([
            os.path.join(self.data_path, f)
            for f in os.listdir(self.data_path)
            if f.endswith(".pkl")
        ])
        self.data = []
        for file in tqdm(self.files, desc="Loading batches"):
            with open(file, "rb") as f:
                batches = pickle.load(f)
                self.data.extend(batches)

        if self.num_samples > 0:
            self.data = self.data[:self.num_samples]

        forbidden = {27, 28, 35, 36}
        self.position_to_id = {}
        current_id = 1
        for r in range(8):
            for c in range(8):
                pos_idx = r * 8 + c
                if pos_idx not in forbidden:
                    pos_name = f"{chr(c + ord('A'))}{r + 1}"
                    self.position_to_id[pos_name] = current_id
                    current_id += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data[index]
        moves = [self.position_to_id[pos] for _, pos in sequence if pos in self.position_to_id]

        board = GameBoard()

        input_ids = [self.bos_token_id] + moves[:-1]
        target_ids = moves[:]

        current_player = Piece.BLACK
        for _, pos in sequence:
            board.add_piece(current_player, pos)
            current_player = Piece.WHITE if current_player == Piece.BLACK else Piece.BLACK

        has_moves_current = len(board.get_legal_moves(current_player)) > 0
        opponent = Piece.BLACK if current_player == Piece.WHITE else Piece.WHITE
        has_moves_opp = len(board.get_legal_moves(opponent)) > 0

        if not has_moves_current and not has_moves_opp:
            if len(target_ids) < self.max_length:
                target_ids.append(self.eos_token_id)
            else:
                target_ids[-1] = self.eos_token_id

        input_ids += [self.pad_token_id] * (self.max_length - len(input_ids))
        target_ids += [self.pad_token_id] * (self.max_length - len(target_ids))

        return torch.tensor(input_ids[:self.max_length], dtype=torch.long), torch.tensor(target_ids[:self.max_length], dtype=torch.long)