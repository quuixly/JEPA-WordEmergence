import os
import pickle
import torch
from tqdm import tqdm

TRAIN_FOLDER = 'train'
OUTPUT_FILE = 'othello_dataset.pt'

ROWS = 'ABCDEFGH'
MAX_MOVES = 60
PADDING = -100
CENTER_POS = {"D4", "E4", "D5", "E5"}


def pos_to_index(pos):
    if pos == PADDING:
        return None
    if isinstance(pos, str):
        if pos in CENTER_POS:
            return None
        row, col = ord(pos[0].upper()) - ord('A'), int(pos[1]) - 1
    elif isinstance(pos, (tuple, list)):
        row, col = pos
        if (row, col) in [(3, 3), (3, 4), (4, 3), (4, 4)]:
            return None
    else:
        raise ValueError(f"Unknown move format: {pos}")

    idx = row * 8 + col
    center_indices = [3 * 8 + 3, 3 * 8 + 4, 4 * 8 + 3, 4 * 8 + 4]
    for c in center_indices:
        if idx > c:
            idx -= 1
    return idx


def process_game(game_moves):
    tensor = torch.full((MAX_MOVES,), PADDING, dtype=torch.int8)
    for i, move in enumerate(game_moves):
        idx = pos_to_index(move)
        if idx is not None and i < MAX_MOVES:
            tensor[i] = idx + 1
    return tensor


def main():
    batch_files = sorted(f for f in os.listdir(TRAIN_FOLDER) if f.endswith('.pkl'))

    total_games = 0
    for batch_file in batch_files:
        with open(os.path.join(TRAIN_FOLDER, batch_file), 'rb') as f:
            batch_data = pickle.load(f)
            total_games += len(batch_data)

    print(f"Total games: {total_games}")

    dataset = torch.full((total_games, MAX_MOVES), PADDING, dtype=torch.int8)

    start_idx = 0
    for batch_file in tqdm(batch_files, desc="Processing batches"):
        with open(os.path.join(TRAIN_FOLDER, batch_file), 'rb') as f:
            batch_data = pickle.load(f)
        for i, game in enumerate(batch_data):
            dataset[start_idx + i] = process_game(game)
        start_idx += len(batch_data)

    print(f"Final dataset shape: {dataset.shape}, dtype: {dataset.dtype}")

    torch.save(dataset, OUTPUT_FILE)
    print(f"Saved dataset to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()