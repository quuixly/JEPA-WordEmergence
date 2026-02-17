# Board Initialization Use Cases

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
---

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

## Saving the Board State

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