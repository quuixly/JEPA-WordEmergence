from enum import Enum

import numpy as np


class Piece(Enum):
    EMPTY = 'E'
    WHITE = 'W'
    BLACK = 'B'

    def __str__(self):
        if self == Piece.WHITE:
            return '\033[97m●\033[0m'
        elif self == Piece.BLACK:
            return '\033[30m●\033[0m'
        else:
            return '·'


class GameBoard:
    __BOARD_SIZE = (8, 8)

    def __init__(self, game_history=None):
        self.game_history = game_history
        self.__board = np.full(GameBoard.__BOARD_SIZE, Piece.EMPTY, dtype=object)

        self.__setup_starting_position()
        self.__restore_game_history()

    def __setup_starting_position(self):
        self.add_piece(Piece.BLACK, 'E4')
        self.add_piece(Piece.BLACK, 'D5')
        self.add_piece(Piece.WHITE, 'E5')
        self.add_piece(Piece.WHITE, 'D4')

    def __restore_game_history(self):
        if self.game_history:
            for move in self.game_history:
                self.add_piece(move[0], move[1])

    def add_piece(self, piece, position):
        if len(position) != 2 or position[0] not in 'ABCDEFGH' or position[1] not in '12345678':
            raise ValueError('Invalid position')
        if piece not in [Piece.BLACK, Piece.WHITE, Piece.EMPTY]:
            raise ValueError('Invalid piece')

        x = int(position[1]) - 1
        y = ord(position[0]) - ord('A')

        self.__board[x, y] = piece

    def __str__(self):
        col_headers = '   ' + '  '.join(chr(ord('A') + i) for i in range(self.__board.shape[1]))
        row_headers = []

        for i, row in enumerate(self.__board):
            row_str = f"{i + 1}  " + '  '.join(str(cell) for cell in row)
            row_headers.append(row_str)

        return '\n'.join([col_headers] + row_headers)


class Othello:
    def __init__(self, game_history = None):
        self.game_history = game_history
        self.board = GameBoard(game_history)
        self.player_turn = Piece.BLACK



if __name__ == '__main__':
    othello = Othello()
    print(othello.board)