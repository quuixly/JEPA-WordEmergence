from enum import Enum

import numpy as np


class Piece(Enum):
    """
    This enum represents the possible values of a tile on the board.
    Using str() or print() will display simple ASCII symbols for easier distinction of tiles.
    """

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
    """
    This class represents the Othello game board. It provides functions for placing pieces
    according to the rules of the game, including ones that do not flip any pieces or
    are placed on illegal positions. This can facilitate future experiments if such functionality
    becomes necessary.
    """

    __BOARD_SIZE = (8, 8)
    # Directions to check for legal moves or opponent pieces to flip
    __DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                    (1, 0), (1, -1), (0, -1), (-1, -1)]

    def __init__(self, game_history=None):
        # History of placed pieces (both legal moves and illegal ones, if such are added)
        self.game_history = game_history if game_history else []
        self.__board = np.full(GameBoard.__BOARD_SIZE, Piece.EMPTY, dtype=object)

        self.__restore_game_history()
        self.__setup_starting_position()

    def get_board(self):
        return self.__board

    def get_game_history(self):
        return self.game_history

    def __setup_starting_position(self):
        """
        This function places the 4 starting pieces in the center of the board,
        which is the standard initial setup for every game.
        """

        self.add_piece(Piece.BLACK, 'E4')
        self.add_piece(Piece.BLACK, 'D5')
        self.add_piece(Piece.WHITE, 'E5')
        self.add_piece(Piece.WHITE, 'D4')

    def __restore_game_history(self):
        """
        This function replays a game history. It should be used for games that were played according to the rules
        (history containing consecutive player moves, assuming pieces are flipped on the board).
        To reconstruct a board based on a list of pieces, use the restore_custom_board function.
        """

        if self.game_history:
            for piece, position in self.game_history:
                # add_piece function without updating game_history
                self.add_piece_without_flip(piece, position)

                if piece != Piece.EMPTY:
                    row, col = self.position_to_index(position)
                    self.__flip_pieces(row, col, piece)
                # ================================================

    def restore_custom_board(self, pieces):
        """
        This function is used to reconstruct a board based on a list of pieces and their positions.
        No intermediate pieces will be flipped (according to the game rules).
        To replay a legal game, pass the history of consecutive pieces
        to the constructor (create a new GameBoard object).

        :param pieces: [[Piece.BLACK, "D4"], [Piece.WHITE, "D5"]]
        """

        if pieces:
            for piece, position in pieces:
                # add_piece_without_flip function without updating game_history
                if len(position) != 2 or position[0] not in 'ABCDEFGH' or position[1] not in '12345678':
                    raise ValueError('Invalid position')
                if piece not in [Piece.BLACK, Piece.WHITE, Piece.EMPTY]:
                    raise ValueError('Invalid piece')

                row, col = self.position_to_index(position)
                self.__board[row, col] = piece
                # =============================================================

    def add_piece_without_flip(self, piece, position):
        """
        This function places a piece on the board without flipping any others, ignoring the game rules.

        :param piece: Piece.WHITE, Piece.BLACK
        :param position: "A4", "B3"
        """

        if len(position) != 2 or position[0] not in 'ABCDEFGH' or position[1] not in '12345678':
            raise ValueError('Invalid position')
        if piece not in [Piece.BLACK, Piece.WHITE, Piece.EMPTY]:
            raise ValueError('Invalid piece')

        row, col = self.position_to_index(position)
        self.__board[row, col] = piece
        self.game_history.append((piece, position))

    def add_piece(self, piece, position):
        """
        This function places a piece on the board following the game rules (flipping the appropriate pieces).

        :param piece: Piece.WHITE, Piece.BLACK
        :param position: "A4", "B3"
        """

        self.add_piece_without_flip(piece, position)

        if piece != Piece.EMPTY:
            row, col = self.position_to_index(position)
            self.__flip_pieces(row, col, piece)
            self.game_history.append((piece, position))

    def __flip_pieces(self, row, col, player):
        """
        This function flips the opponent's pieces after placing a piece on the board.

        :param row:  int, row of placed piece
        :param col: int, col of placed piece
        :param player: Piece.BLACK, Piece.WHITE
        """

        flips = self.__get_flippable_pieces(row, col, player, for_move_check=False)

        for x, y in flips:
            self.__board[x, y] = player

    def get_legal_moves(self, player):
        """
        This function returns all possible legal moves for a player according to the game rules.

        :param player: Piece.BLACK, Piece.WHITE
        :return: ["A4", "B3"]
        """

        legal_moves = []

        for row in range(GameBoard.__BOARD_SIZE[0]):
            for col in range(GameBoard.__BOARD_SIZE[1]):
                if self.__board[row, col] != Piece.EMPTY:
                    continue

                if self.__get_flippable_pieces(row, col, player, for_move_check=True):
                    legal_moves.append(self.index_to_position((row, col)))

        return legal_moves

    def __get_flippable_pieces(self, row, col, player, for_move_check=False):
        """
        This function checks which opponent pieces would be flipped if the player places a piece at (row, col).

        :param row: int, row index where the player intends to place a piece
        :param col: int, col index where the player intends to place a piece
        :param player: Piece.BLACK, Piece.WHITE
        :param for_move_check: bool, if True, only checks move legality and returns True/False
        :return: list of (row, col) tuples of opponent pieces to flip if for_move_check is False;
         bool indicating move legality if for_move_check is True
        """

        opponent = Piece.BLACK if player == Piece.WHITE else Piece.WHITE
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
        """
        This function converts a board tile index into its string representation.

        :param index: (int, int)
        :return: "A4", "B3"
        """

        row, col = index

        row = int(row)
        col = int(col)

        return chr(col + ord('A')) + str(row + 1)

    @staticmethod
    def position_to_index(position):
        """
        This function converts a board position from its string representation to a numeric index.

        :param position: "A4", "B3"
        :return: (int, int)
        """

        if len(position) != 2 or position[0] not in 'ABCDEFGH' or position[1] not in '12345678':
            raise ValueError('Invalid position')

        row = int(position[1]) - 1
        col = ord(position[0].upper()) - ord('A')

        return row, col

    def __str__(self):
        """
        Returns a string representation of the board, with column headers as letters (A-H)
        and row numbers (1-8), showing the current state of each tile.
        """

        col_headers = '   ' + '  '.join(chr(ord('A') + i) for i in range(GameBoard.__BOARD_SIZE[1]))
        rows = []

        for i, row in enumerate(self.__board):
            row_str = f"{i + 1}  " + '  '.join(str(cell) for cell in row)
            rows.append(row_str)

        return '\n'.join([col_headers] + rows)

    def display(self, highlight_positions=None):
        """
        Displays the board in the console with optional highlighting. Highlights are displayed
        with a red background in the terminal.

        :param highlight_positions: list of board positions (e.g., "D3") to highlight
        """

        highlight_positions = highlight_positions or []

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
    """
    This class was created to play Othello. It allows players to gain intuition about how the game works,
    and also to play using a specific board configuration for future research purposes.
    """

    def __init__(self, game_history=None):
        self.game_history = game_history
        self.board = GameBoard(game_history)
        self.player_turn = self.determine_starting_player()

    def determine_starting_player(self):
        """
        Determines which player should move next.

        - If the game has just started (no history), Black moves first.
        - Otherwise, returns the opposite of the last played piece.

        :return: Piece.BLACK or Piece.WHITE
        """

        if not self.game_history:
            return Piece.BLACK

        last_piece = self.game_history[-1][0]

        return Piece.BLACK if last_piece == Piece.WHITE else Piece.WHITE

    def display(self, highlight_positions=None):
        """
        Displays the current board state, optionally highlighting specific positions.

        :param highlight_positions: list of positions "A3", "D3" to highlight on the board
        """

        self.board.display(highlight_positions)

    def get_legal_moves(self):
        """
        Returns all legal moves for the current player.

        :return: list of positions "A3", "D1" where the current player can legally place a piece
        """

        return self.board.get_legal_moves(self.player_turn)

    def switch_turn(self):
        """
        Switches the turn to the other player.
        If the current turn is WHITE, it changes to BLACK, and vice versa.
        """

        self.player_turn = Piece.BLACK if self.player_turn == Piece.WHITE else Piece.WHITE

    def input_move(self, legal_moves):
        """
        Prompts the player to enter a move until a valid one is provided.

        :param legal_moves: list of strings "A2", "D1" representing the moves allowed for the current player
        :return: the move chosen by the player (as a string)
        """

        while True:
            move = input(f"Enter your move ({', '.join(legal_moves)}): ").upper()

            if move in legal_moves:
                return move

            print("Invalid move. Try again.")

    def check_game_over(self):
        """
        Checks whether the game is over.
        The game is over if neither the current player nor the opponent has any legal moves.

        :return: True if the game has ended, False otherwise
        """

        current_moves = self.get_legal_moves()

        opponent = Piece.BLACK if self.player_turn == Piece.WHITE else Piece.WHITE
        opponent_moves = self.board.get_legal_moves(opponent)

        return not current_moves and not opponent_moves

    def print_result(self):
        """
        Prints the final result of the game, including the score and the winner.

        Counts the number of black and white pieces on the board and prints:
        - The final score
        - Which player won, or if it's a tie
        """

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
        """
        Main game loop for playing Othello.

        - Prints the current player's turn.
        - Shows the board with legal moves highlighted.
        - If the current player has no legal moves, skips the turn.
        - Prompts the player to input a move and updates the board.
        - Switches turns after each valid move.
        - Continues until the game is over.
        - At the end, displays the final board and prints the game result.
        """

        while True:
            print(f"\nCurrent turn: {'Black' if self.player_turn == Piece.BLACK else 'White'}")
            legal_moves = self.get_legal_moves()
            self.display(legal_moves)

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