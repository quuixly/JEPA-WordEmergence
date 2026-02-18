from game.othello import GameBoard, Piece
import random
from tqdm import tqdm
import hashlib
import os
import pickle
import multiprocessing
import psutil
import time


def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def worker_game_generator(queue, seed):
    random.seed(seed)

    while True:
        try:
            board = GameBoard()
            current_player = Piece.BLACK
            passed_previously = False

            while True:
                legal_moves = board.get_legal_moves(current_player)
                if not legal_moves:
                    if passed_previously:
                        break
                    passed_previously = True
                else:
                    passed_previously = False
                    move = random.choice(legal_moves)
                    board.add_piece(current_player, move)

                current_player = Piece.BLACK if current_player == Piece.WHITE else Piece.WHITE

            queue.put(board.get_game_history())

        except BrokenPipeError:
            break
        except Exception as e:
            print(f"Worker error: {e}")
            break


class OthelloDatasetGenerator:
    @staticmethod
    def _hash_sequence(seq):
        s = str(seq).encode()
        return hashlib.sha256(s).hexdigest()

    @staticmethod
    def _save_batch(data_batch, folder, batch_index):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f'batch_{batch_index:04d}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(data_batch, f)

    @staticmethod
    def generate(n_train, n_test, batch_size=500_000, train_folder='train', test_folder='test'):
        num_workers = os.cpu_count()
        if num_workers is None: num_workers = 4

        queue = multiprocessing.Queue(maxsize=1000)

        workers = []
        print(f"Number of workers: {num_workers}")

        for i in range(num_workers):
            seed = random.randint(0, 10 ** 9) + i
            p = multiprocessing.Process(target=worker_game_generator, args=(queue, seed))
            p.daemon = True
            p.start()
            workers.append(p)

        try:
            seen_games_hashes = set()
            train_subsequences_lookup = set()

            def get_next_unique_game_from_queue():
                while True:
                    full_history = queue.get()

                    game_hash = OthelloDatasetGenerator._hash_sequence(full_history)
                    if game_hash not in seen_games_hashes:
                        seen_games_hashes.add(game_hash)
                        return full_history

            train_data = []
            batch_index = 0

            print(f"Train Set ({n_train})...")
            with tqdm(total=n_train) as pbar:
                while (batch_index * batch_size) + len(train_data) < n_train:
                    full_history = get_next_unique_game_from_queue()

                    if len(full_history) == 0: continue
                    k = random.randint(1, len(full_history))
                    subsequence = full_history[:k]

                    subseq_hash = OthelloDatasetGenerator._hash_sequence(subsequence)
                    train_subsequences_lookup.add(subseq_hash)

                    train_data.append(subsequence)
                    pbar.update(1)
                    pbar.set_postfix({'RAM_MB': f"{get_memory_mb():.1f}"})

                    if len(train_data) >= batch_size:
                        OthelloDatasetGenerator._save_batch(train_data, train_folder, batch_index)
                        train_data = []
                        batch_index += 1

            if train_data:
                OthelloDatasetGenerator._save_batch(train_data, train_folder, batch_index)
                train_data = []

            test_data = []
            batch_index = 0

            print(f"Test Set ({n_test})...")
            with tqdm(total=n_test) as pbar:
                while (batch_index * batch_size) + len(test_data) < n_test:
                    full_history = get_next_unique_game_from_queue()

                    if len(full_history) == 0: continue
                    k = random.randint(1, len(full_history))
                    subsequence = full_history[:k]
                    subseq_hash = OthelloDatasetGenerator._hash_sequence(subsequence)

                    if subseq_hash in train_subsequences_lookup:
                        continue

                    test_data.append(subsequence)
                    pbar.update(1)
                    pbar.set_postfix({'RAM_MB': f"{get_memory_mb():.1f}"})

                    if len(test_data) >= batch_size:
                        OthelloDatasetGenerator._save_batch(test_data, test_folder, batch_index)
                        test_data = []
                        batch_index += 1

            if test_data:
                OthelloDatasetGenerator._save_batch(test_data, test_folder, batch_index)

        finally:
            for p in workers:
                p.terminate()
                p.join()


if __name__ == '__main__':
    N_TRAIN = 20_000_000
    N_TEST = 1_000_000
    BATCH_SIZE = 500_000

    OthelloDatasetGenerator.generate(
        N_TRAIN, N_TEST,
        batch_size=BATCH_SIZE,
        train_folder='train',
        test_folder='test'
    )