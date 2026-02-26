from game.othello import GameBoard, Piece
import random
from tqdm import tqdm
import hashlib
import os
import pickle
import multiprocessing
import itertools


def generate_single_game(seed):
    rng = random.Random(seed)
    board = GameBoard()
    current_player = Piece.BLACK
    passed_previously = False

    history = []

    while True:
        legal_moves = board.get_legal_moves(current_player)
        if not legal_moves:
            if passed_previously:
                break
            passed_previously = True
        else:
            passed_previously = False
            move = rng.choice(legal_moves)
            board.add_piece(current_player, move)

            history.append(move)

        current_player = Piece.BLACK if current_player == Piece.WHITE else Piece.WHITE

    return history


class OthelloDatasetGenerator:
    @staticmethod
    def _hash_sequence(seq):
        return hashlib.sha1(pickle.dumps(seq)).digest()

    @staticmethod
    def _save_batch(data_batch, folder, batch_index):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f'batch_{batch_index:04d}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(data_batch, f)

    @staticmethod
    def generate(n_train, batch_size=500_000, train_folder='train'):
        num_workers = os.cpu_count() or 4
        seen_games_hashes = set()
        train_data = []
        batch_index = 0

        with multiprocessing.Pool(num_workers) as pool:
            seeds = itertools.count()

            with tqdm(total=n_train) as pbar:
                for full_history in pool.imap_unordered(generate_single_game, seeds, chunksize=500):

                    if not full_history:
                        continue

                    game_hash = OthelloDatasetGenerator._hash_sequence(full_history)
                    if game_hash in seen_games_hashes:
                        continue

                    seen_games_hashes.add(game_hash)

                    padded_game = list(full_history)
                    if len(padded_game) < 60:
                        padded_game += [-100] * (60 - len(padded_game))

                    train_data.append(padded_game)
                    pbar.update(1)

                    if len(train_data) >= batch_size:
                        OthelloDatasetGenerator._save_batch(train_data, train_folder, batch_index)
                        train_data = []
                        batch_index += 1

                    if len(seen_games_hashes) >= n_train:
                        break

        if train_data:
            OthelloDatasetGenerator._save_batch(train_data, train_folder, batch_index)



if __name__ == '__main__':
    N_TRAIN = 20_000_000
    BATCH_SIZE = 500_000

    OthelloDatasetGenerator.generate(
        N_TRAIN,
        batch_size=BATCH_SIZE,
        train_folder='train'
    )