# Creating Custom Board States

You can construct board states in two different ways depending on your needs.

Remember that **the board always starts with 4 pieces** in the center according to the game rules.
These initial pieces are **not included in game_history**. If you want **to remove any of them from the board**, simply **place a piece on one of those squares** **(it will overwrite the existing piece and add to game_history)**.

## 1 Creating a Board Using `add_piece` (With Flipping – Legal Moves)

Use `add_piece()` when you want to simulate a real, legal Othello game.
This method follows the game rules and automatically flips opponent pieces.

```python
from othello import GameBoard, Piece

board = GameBoard()

board.add_piece(Piece.BLACK, "D3")
board.add_piece(Piece.WHITE, "C3")
```

* Flipping is applied automatically.
* The move is stored in `game_history`.
* The board behaves exactly like a real game.
* No legality checks are enforced. To know where a piece can be placed, use:
```python
from othello import GameBoard, Piece

board = GameBoard()

moves = board.get_legal_moves(Piece.WHITE)
board.add_piece(Piece.BLACK, moves[0]) # Example
```

Use this for:

* Simulating games
* Generating training data from legal play
* Replaying historical matches

## 2 Creating a Board Using `add_piece_without_flip` (No Rule Enforcement)

Use `add_piece_without_flip()` when you want full control over the board state.

```python
board = GameBoard()

board.add_piece_without_flip(Piece.BLACK, "A1")
board.add_piece_without_flip(Piece.WHITE, "H8")
```

* No flipping is performed.
* No legality checks are enforced.
* The move is stored in `game_history`.
* Pieces are placed exactly as specified.

Use this for:

* Creating artificial board states
* Research experiments
* Testing edge cases
* Constructing partially invalid positions intentionally

---

# Board Initialization

This implementation supports two different ways of reconstructing a board:

## 1. Reconstructing a Board from Legal Move History

Use this when you have a sequence of **legal moves played in order**.

Each move must follow the game rules and will automatically flip opponent pieces.

### Example

```python
from othello import GameBoard, Piece

game_history = [
    (Piece.BLACK, "D3"),
    (Piece.WHITE, "C3"),
    (Piece.BLACK, "C4"),
    (Piece.WHITE, "F5"),
]

board = GameBoard(game_history)
board.display()
```

or if you want to play:

```python
from othello import Othello, Piece

game_history = [
    (Piece.BLACK, "D3"),
    (Piece.WHITE, "C3"),
    (Piece.BLACK, "C4"),
    (Piece.WHITE, "F5"),
]

game = Othello(game_history)
game.play()
```

Use this approach when:

* Replaying real games
* Continuing a previously (legal) played game

## 2 Reconstructing a Board from a List of Pieces (Custom State)

Use this when you want to directly specify which pieces are on which tiles.

No flipping is performed.

This ignores Othello rules and directly sets the board state.

### Example

```python
from othello import GameBoard, Piece

board = GameBoard()

custom_pieces = [
    (Piece.BLACK, "A1"),
    (Piece.BLACK, "H8"),
    (Piece.WHITE, "D4"),
    (Piece.WHITE, "E5"),
]

board.restore_custom_board(custom_pieces)
board.display()
```

or if you want to play:

```python
from othello import Othello, Piece

game_history = [
    (Piece.BLACK, "D3"),
    (Piece.WHITE, "C3"),
    (Piece.BLACK, "C4"),
    (Piece.WHITE, "F5"),
]

game = Othello()
game.board.restore_custom_board(game_history)
game.play()
```

Use this approach when:

* Running experiments
* Creating artificial positions

---

# Saving the Board State

The board state can be saved using the `get_game_history()` method from the `GameBoard` class.

This method returns a list containing all pieces that were added to the board, along with their positions:

```python
game_history = othello.board.get_game_history()
```

The returned format is:

```python
[(Piece.BLACK, "D3"), (Piece.WHITE, "C3"), ...]
```

This list contains **all pieces that were added to the board**, in the order they were placed.