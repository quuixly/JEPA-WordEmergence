import os
import pickle
import numpy as np
from tqdm import tqdm

TRAIN_FOLDER = 'train'
OUTPUT_FILE = 'othello_dataset.npy'

def load_batches(folder):
    batch_files = sorted(f for f in os.listdir(folder) if f.endswith('.pkl'))
    all_games = []

    for batch_file in tqdm(batch_files, desc="Loading batches"):
        path = os.path.join(folder, batch_file)
        with open(path, 'rb') as f:
            batch_data = pickle.load(f)
            all_games.extend(batch_data)

    return all_games

def main():
    all_games = load_batches(TRAIN_FOLDER)
    print(f"Loaded {len(all_games)} games.")

    dataset = np.array(all_games, dtype=np.int8)
    print(f"Dataset shape: {dataset.shape}, dtype: {dataset.dtype}")

    np.save(OUTPUT_FILE, dataset)
    print(f"Saved dataset to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()