from chess_tournament import Game, Player, RandomPlayer
import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from chess_tournament.players import Player


class TransformerPlayer(Player):
    """
    Transformer-based chess player with rule-based filters.

    Architecture:
        1. _critical_move()   — code: checkmate in 1, pawn promotion (never miss instant wins)
        2. model (x3 retries) — transformer: Stockfish-trained strategic move selection
        3. _is_safe_move()    — code: reject model moves that immediately hang a piece
        4. _heuristic_move()  — code: safety net if model completely fails

    REQUIRED:
        Subclasses chess_tournament.players.Player
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    # piece values
    PIECE_VALUES = {
        chess.QUEEN: 9, chess.ROOK: 5,
        chess.BISHOP: 3, chess.KNIGHT: 3, chess.PAWN: 1
    }

    def __init__(
        self,
        name: str = "TinyLMPlayer",
        model_id: str = "EvelienUU/chess-qwen-finetuned-v2",
        temperature: float = 0.1,
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
        board = chess.Board(fen)
        legal_moves = [move.uci() for move in board.legal_moves]
        legal_moves_str = ", ".join(legal_moves)

        return f"""You are an expert chess player.
You are given a chess position in FEN notation and a list of legal moves.
Your goal is to win the game by making strong, strategic moves.
Analyze the position and think about the consequences of each move before deciding.
Consider: which pieces are under attack, what your opponent might do next, and which move improves your position the most.
Choose the best legal move from the legal moves list.
Output the only the move in UCI format (example: e2e4), nothing else.

FEN: {fen}
Legal moves: {legal_moves_str}
Best move:"""

    def _extract_move(self, text: str) -> Optional[str]:
        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    # -------------------------
    # Rule-based filters
    # -------------------------
    def _is_safe_move(self, board: chess.Board, move: chess.Move) -> bool:
        """
        Returns True if the piece we just moved won't be immediately captured.
        """
        moving_piece = board.piece_at(move.from_square)
        if moving_piece is None:
            return True

        board.push(move)  # temporarily making the move
        to_square = move.to_square

        # checking if opponent can capture on the square we just moved to
        if board.is_attacked_by(board.turn, to_square):
            # finding the lowest value attacker the opponent has
            min_attacker_value = 99
            for attacker_sq in board.attackers(board.turn, to_square):
                attacker = board.piece_at(attacker_sq)
                if attacker:
                    val = self.PIECE_VALUES.get(attacker.piece_type, 0)
                    if val < min_attacker_value:
                        min_attacker_value = val

            our_piece_value = self.PIECE_VALUES.get(moving_piece.piece_type, 0)

            board.pop()

            # unsafe if opponent can capture our piece with something cheaper
            if min_attacker_value < our_piece_value:
                return False

            return True

        board.pop()
        return True

    # --- ADDED: checks if opponent has no pieces left besides king and pawns ---
    def _is_endgame(self, board: chess.Board) -> bool:
        """
        Endgame = opponent has no major or minor pieces left.
        In this case the model has no endgame technique so we use dedicated logic.
        """
        opponent = not board.turn
        return sum(
            len(board.pieces(pt, opponent))
            for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        ) == 0

    # --- ADDED: picks the move that restricts the opponent king the most ---
    def _endgame_move(self, board: chess.Board) -> Optional[str]:
        """
        Instead of shuffling pieces randomly in a winning endgame, this picks
        the move that leaves the opponent king with the fewest legal moves.
        Also explicitly avoids stalemate — that would throw away a won position.
        """
        best_move, best_score = None, -999

        for move in board.legal_moves:
            if not self._is_safe_move(board, move):
                continue

            board.push(move)

            # take checkmate immediately
            if board.is_checkmate():
                board.pop()
                return move.uci()

            # never stalemate the opponent — that's a draw when we're winning
            if board.is_stalemate():
                board.pop()
                continue

            # fewer opponent moves = more restricted king = better for us
            score = -len(list(board.legal_moves))

            # bonus for giving check — restricts the king right away
            if board.is_check():
                score += 5

            board.pop()

            if score > best_score:
                best_score, best_move = score, move

        return best_move.uci() if best_move else None

    def _critical_move(self, fen: str) -> Optional[str]:
        """
        Always checked before the model to never miss obvious wins.
        """
        board = chess.Board(fen)

        # checkmate in 1
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move.uci()
            board.pop()

        # pawn promotion to queen
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN:
                rank = chess.square_rank(move.to_square)
                if (piece.color == chess.WHITE and rank == 7) or \
                   (piece.color == chess.BLACK and rank == 0):
                    promotion_move = chess.Move(
                        move.from_square, move.to_square, promotion=chess.QUEEN
                    )
                    if promotion_move in board.legal_moves:
                        return promotion_move.uci()

        # --- ADDED: use endgame logic instead of the model when opponent has no pieces ---
        if self._is_endgame(board):
            return self._endgame_move(board)

        return None

    def _heuristic_move(self, fen: str) -> Optional[str]:
        """
        The safety net: only used if the model fails.
        """
        board = chess.Board(fen)

        # best capture
        best_capture = None
        best_value = 0
        for move in board.legal_moves:
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured and self.PIECE_VALUES.get(captured.piece_type, 0) > best_value:
                    best_value = self.PIECE_VALUES[captured.piece_type]
                    best_capture = move
        if best_capture:
            return best_capture.uci()

        # castling (king safety, first 20 moves only)
        if board.fullmove_number <= 20:
            for move in board.legal_moves:
                if board.is_castling(move):
                    return move.uci()

        # give check
        for move in board.legal_moves:
            board.push(move)
            if board.is_check():
                board.pop()
                return move.uci()
            board.pop()

        # random legal move
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:

        # Step 1: Critical moves
        critical = self._critical_move(fen)
        if critical:
            return critical

        # Step 2: Loading the model
        try:
            self._load_model()
        except Exception:
            return self._heuristic_move(fen)

        # Step 3: Asking the model, up to 3 retries
        prompt = self._build_prompt(fen)

        try:
            board = chess.Board(fen)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            for _ in range(3):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=self.temperature > 0,
                        temperature=self.temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                if decoded.startswith(prompt):
                    decoded = decoded[len(prompt):]

                move_str = self._extract_move(decoded)

                if move_str:
                    try:
                        move = chess.Move.from_uci(move_str)
                        if move in board.legal_moves:
                            # Step 4: Safety filter
                            if self._is_safe_move(board, move):
                                return move_str
                            # Move is legal but unsafe: keep retrying
                    except ValueError:
                        pass

        except Exception:
            pass

        # Step 5: Model failed or kept suggesting unsafe moves: heuristics
        return self._heuristic_move(fen)
