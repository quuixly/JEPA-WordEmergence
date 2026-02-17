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
    __DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                    (1, 0), (1, -1), (0, -1), (-1, -1)]

    def __init__(self, game_history=None):
        self.game_history = game_history
        self.__board = np.full(GameBoard.__BOARD_SIZE, Piece.EMPTY, dtype=object)

        self.__setup_starting_position()
        self.__restore_game_history()

    def get_board(self):
        return self.__board

    def __setup_starting_position(self):
        self.add_piece(Piece.BLACK, 'E4')
        self.add_piece(Piece.BLACK, 'D5')
        self.add_piece(Piece.WHITE, 'E5')
        self.add_piece(Piece.WHITE, 'D4')

    def __restore_game_history(self):
        if self.game_history:
            for piece, position in self.game_history:
                self.add_piece(piece, position)

    def restore_custom_board(self, pieces):
        if pieces:
            for piece, position in pieces:
                self.add_piece_without_flip(piece, position)

    def add_piece_without_flip(self, piece, position):
        if len(position) != 2 or position[0] not in 'ABCDEFGH' or position[1] not in '12345678':
            raise ValueError('Invalid position')
        if piece not in [Piece.BLACK, Piece.WHITE, Piece.EMPTY]:
            raise ValueError('Invalid piece')

        row, col = self.position_to_index(position)
        self.__board[row, col] = piece

    def add_piece(self, piece, position):
        self.add_piece_without_flip(piece, position)

        if piece != Piece.EMPTY:
            row, col = self.position_to_index(position)
            self.__flip_pieces(row, col, piece)

    def __flip_pieces(self, row, col, player):
        opponent = Piece.BLACK if player == Piece.WHITE else Piece.WHITE
        flips = self.__get_flippable_pieces(row, col, player, opponent, for_move_check=False)

        for x, y in flips:
            self.__board[x, y] = player

    def get_legal_moves(self, player):
        opponent = Piece.BLACK if player == Piece.WHITE else Piece.WHITE
        legal_moves = []

        for row in range(GameBoard.__BOARD_SIZE[0]):
            for col in range(GameBoard.__BOARD_SIZE[1]):
                if self.__board[row, col] != Piece.EMPTY:
                    continue

                if self.__get_flippable_pieces(row, col, player, opponent, for_move_check=True):
                    legal_moves.append(self.index_to_position((row, col)))

        return legal_moves

    def __get_flippable_pieces(self, row, col, player, opponent, for_move_check=False):
        flippable_total = []

        for dx, dy in GameBoard.__DIRECTIONS:
            x, y = row + dx, col + dy
            flips = []

            while 0 <= x < GameBoard.__BOARD_SIZE[0] and 0 <= y < GameBoard.__BOARD_SIZE[1]:
                if self.__board[x, y] == opponent:
                    flips.append((x, y))
                elif self.__board[x, y] == player:
                    if flips:
                        if for_move_check:
                            return True
                        flippable_total.extend(flips)
                    break
                else:
                    break

                x += dx
                y += dy

        if for_move_check:
            return False

        return flippable_total

    @staticmethod
    def index_to_position(index):
        row, col = index

        row = int(row)
        col = int(col)

        return chr(col + ord('A')) + str(row + 1)

    @staticmethod
    def position_to_index(position):
        if len(position) != 2 or position[0] not in 'ABCDEFGH' or position[1] not in '12345678':
            raise ValueError('Invalid position')

        row = int(position[1]) - 1
        col = ord(position[0].upper()) - ord('A')

        return row, col

    def __str__(self):
        col_headers = '   ' + '  '.join(chr(ord('A') + i) for i in range(GameBoard.__BOARD_SIZE[1]))
        rows = []

        for i, row in enumerate(self.__board):
            row_str = f"{i + 1}  " + '  '.join(str(cell) for cell in row)
            rows.append(row_str)

        return '\n'.join([col_headers] + rows)

    def display(self, highlight_positions=None, possible_moves_for_player=None):
        highlight_positions = highlight_positions or []
        if possible_moves_for_player:
            legal_moves = self.get_legal_moves(possible_moves_for_player)
            highlight_positions += legal_moves

        highlight_tuples = [GameBoard.position_to_index(pos) for pos in highlight_positions]
        board_str = str(self)
        lines = board_str.split('\n')

        for i in range(1, len(lines)):
            row_num_str = lines[i][:3]
            row_content = lines[i][3:]
            cells = [c for c in row_content.split('  ') if c]
            new_row = ''
            for j, cell in enumerate(cells):
                if (i - 1, j) in highlight_tuples:
                    new_row += '\033[41m' + cell + '\033[0m  '  # Red background for highlights
                else:
                    new_row += cell + '  '
            lines[i] = row_num_str + new_row.rstrip()

        print('\n'.join(lines))


class Othello:
    def __init__(self, game_history=None):
        self.game_history = game_history
        self.board = GameBoard(game_history)
        self.player_turn = self.determine_starting_player()

    def determine_starting_player(self):
        if not self.game_history:
            return Piece.BLACK

        last_piece = self.game_history[-1][0]

        return Piece.BLACK if last_piece == Piece.WHITE else Piece.WHITE

    def display(self, highlight_positions=None, possible_moves_for_player=None):
        self.board.display(highlight_positions, possible_moves_for_player)

    def get_legal_moves(self):
        return self.board.get_legal_moves(self.player_turn)

    def switch_turn(self):
        self.player_turn = Piece.BLACK if self.player_turn == Piece.WHITE else Piece.WHITE

    def input_move(self, legal_moves):
        while True:
            move = input(f"Enter your move ({', '.join(legal_moves)}): ").upper()

            if move in legal_moves:
                return move

            print("Invalid move. Try again.")

    def check_game_over(self):
        current_moves = self.get_legal_moves()

        opponent = Piece.BLACK if self.player_turn == Piece.WHITE else Piece.WHITE
        opponent_moves = self.board.get_legal_moves(opponent)

        return not current_moves and not opponent_moves

    def print_result(self):
        board = self.board.get_board()
        black_count = np.sum(board == Piece.BLACK)
        white_count = np.sum(board == Piece.WHITE)

        print(f"\nFinal Score: Black {black_count} - White {white_count}")

        if black_count > white_count:
            print("Black wins!")
        elif white_count > black_count:
            print("White wins!")
        else:
            print("It's a tie!")

    def play(self):
        while True:
            print(f"\nCurrent turn: {'Black' if self.player_turn == Piece.BLACK else 'White'}")
            self.display(possible_moves_for_player=self.player_turn)

            legal_moves = self.get_legal_moves()
            if not legal_moves:
                print(f"No legal moves for {'Black' if self.player_turn == Piece.BLACK else 'White'}. Skipping turn.")
                self.switch_turn()

                if self.check_game_over():
                    break

                continue

            move = self.input_move(legal_moves)
            self.board.add_piece(self.player_turn, move)
            self.switch_turn()

        self.display()
        self.print_result()


if __name__ == '__main__':
    othello = Othello()
    othello.play()