from chess_tournament import Game, Player, RandomPlayer
import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

class TransformerPlayer(Player):
    """
    Tiny LM baseline chess player.

    REQUIRED:
        Subclasses chess_tournament.players.Player
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "TinyLMPlayer",
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", #Qwen/Qwen2.5-1.5B-Instruct" #"HuggingFaceTB/SmolLM2-135M-Instruct"
        temperature: float = 0.7,
        max_new_tokens: int = 8,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy-loaded components
        self.tokenizer = None
        self.model = None

    # -------------------------
    # Lazy loading
    # -------------------------
    def _load_model(self):
        if self.model is None:
            print(f"[{self.name}] Loading {self.model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

    # -------------------------
    # Prompt
    # -------------------------
    def _build_prompt(self, fen: str) -> str:
        # Giving the board info and legal moves to the prompt in advance so that it can pick from the legal options
        board = chess.Board(fen)
        legal_moves = [move.uci() for move in board.legal_moves]
        legal_moves_str = ", ".join(legal_moves)

        return f"""You are an expert chess player. 
        You are given a chess position in FEN notation and a list of legal moves. 
        Choose the best legal move from the legal moves list.
        Output the only the move in UCI format (example: e2e4), nothing else.

        FEN: {fen}
        Legal moves: {legal_moves_str}
        Best move:"""

    def _extract_move(self, text: str) -> Optional[str]:
        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    def _random_legal(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:

        try:
            self._load_model()
        except Exception:
            return self._random_legal(fen)

        prompt = self._build_prompt(fen)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):]

            move = self._extract_move(decoded)

            if move:
              try: # added this
                if chess.Move.from_uci(move) in chess.Board(fen).legal_moves: # added this
                  return move
              except: # added this
                pass # added this

        except Exception:
            pass

        return self._random_legal(fen)
