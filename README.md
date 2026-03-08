# Chess Transformer Player

Return: UCI move string (e2e4) OR None

## Installation

Must install the instructor package. In Colab:

```
git clone https://github.com/bylinina/chess_exam.git
cd chess-exam
pip install -e .
```

## Example

```python
from player import TransformerPlayer

p = TransformerPlayer("MyBot")
move = p.get_move("starting FEN")
print(move)
```
