# Chess Transformer Player

Return: UCI move string (e2e4) OR None

## Model
- Base model: Qwen2.5-0.5B-Instruct
- Finetuned on: aicrowd/ChessExplained dataset (100k examples total)
- Model on HuggingFace: EvelienUU/chess-qwen-finetuned-v2

## Architecture

My player combines the fine-tuned model with rule-based heuristics. The way the player chooses a move goes like this:
1.	Critical moves 
- checkmate
- pawn promotion to queen

If game is in Endgame (opponent has 1 or zero pieces left besides pawns and a king), other critical moves are considered in order of priority:
- checkmate
- A move that minimizes the opponent's king’s available moves
The endgame function skips unsafe moves, skips stalemate moves, and skips repetition.

2.	Model chooses
If no critical move is found, the fine-tuned model selects a move from the legal moves list. The model gets up to 3 retries. A chosen move is checked by a safety filter: a move is rejected if the opponent can immediately recapture our piece with a cheaper piece. If the model suggested 3 unsafe moves, the heuristics fallback takes over.

3.	Heuristics
When the model has not been able to select a safe move, additional heuristics are used. In order of priority: first, the best capture (highest value opponent piece). Second, castle when in opening. Third, giving check. Fourth: random legal move.

With this approach, I also verified that the model still makes most of the moves in a game. The heuristics only act as filter and fallback.


## Installation

Must install the instructor package. In Colab:

```
git clone https://github.com/bylinina/chess_exam.git
cd chess-exam
pip install -e .
```


## Usage

```python
from player import TransformerPlayer
player = TransformerPlayer(name="EvelienvanDriel")
```
