from chess_tournament import Game, Player, RandomPlayer
import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

class TransformerPlayer(Player):
    """
    Transformer-based chess player with rule-based filters.

    Architecture:
        1. _critical_move()     before using model, check: checkmate in 1, pawn promotion to queen
        2. model (x3 retries)   move selection by finetuned model
        3. _safe_move()         rejecting model moves that lead to capturing of own pieces
        4. _heuristic_move()    safety-net if the model fails

    REQUIRED:
        Subclasses chess_tournament.players.Player
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "TinyLMPlayer",
        model_id: str = "EvelienUU/chess-qwen-finetuned-v2", # finetuned model from HuggingFace
        temperature: float = 0.1, # lowered the temp
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

        # seen positions
        self.seen_positions = []  # list of the seen positions
        self.seen_positions_window = 6  # remembering last 6 positions


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
        """
        Here, i added the legal moves to the prompt of the model. The model was finetuned on this same prompt.
        """
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


    # -------------------------
    # Extracting the move
    # -------------------------
    def _extract_move(self, text: str) -> Optional[str]:
        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None


    # -------------------------
    # Filters
    # -------------------------
    piece_values = {chess.QUEEN: 9, chess.ROOK: 5, chess.BISHOP: 3, chess.KNIGHT: 3, chess.PAWN: 1}

    def _safe_move(self, board: chess.Board, move: chess.Move) -> bool:
        """
        Returns True if the piece we just moved won't be immediately captured.
        """
        moving_piece = board.piece_at(move.from_square) # selecting the piece i want to move
        if moving_piece is None:
            return True

        board.push(move)  # temporarily making the move
        to_square = move.to_square

        # checking if opponent can capture on the square we just moved to
        if board.is_attacked_by(board.turn, to_square):
            # finding the lowest value piece the opponent can attack with
            min_attacker_value = 99
            # looping over all attackers
            for attacker_sq in board.attackers(board.turn, to_square):
                attacker = board.piece_at(attacker_sq) # selecting attacker
                if attacker:
                    val = self.piece_values.get(attacker.piece_type, 0) # getting the piece value
                    if val < min_attacker_value:
                        min_attacker_value = val

            our_piece_value = self.piece_values.get(moving_piece.piece_type, 0) # value of the piece we moved

            board.pop() # undo move

            # I want it to be "unsafe" if the opponent can capture our piece with something cheaper
            if min_attacker_value < our_piece_value:
                return False

            return True # safe

        board.pop()
        return True # if it is not attacked at all

    """
    I saw patterns in the game log of pieces moving back and forth when the opponent only had his king (and a only few other pieces) left, causing a draw.
    Because of this, i defined endgame functions below.
    """

    def _endgame(self, board: chess.Board) -> bool:
        """
        I defined endgame as: opponent has one or zero pieces left besides pawns and a king.
        Function:
        - checks how many pieces the opponent has left
        """
        opponent = not board.turn
        return sum(
            len(board.pieces(pt, opponent))
            for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        ) <= 1


    def _endgame_move(self, board: chess.Board) -> Optional[str]:
        """
        Endgame strategy.
        This function:
        -  picks the move that leaves the opponents king with the lowest amount of legal moves.
        -  avoids stalemate.
        """
        best_move, best_score = None, -999 # tracking the best move this far

        # looping over all legal moves
        for move in board.legal_moves:
            if not self._safe_move(board, move): # safe move yes or no?
                continue

            board.push(move)

            # take checkmate if it is checkmate
            if board.is_checkmate():
                board.pop()
                return move.uci()

            # never stalemate the opponent
            if board.is_stalemate():
                board.pop()
                continue

            # dont want repetition of moves (i saw lots of repeating steps in the game logs)
            position_key = board.fen().split(' ')[0] # the piece positions
            if position_key in self.seen_positions: # skipping if we did thid move in the 6 latest moves
                board.pop()
                continue

            # fewer opponent moves: i want the opponent to have as little moves as possible
            score = -len(list(board.legal_moves))

            # bonus points if move gives check
            if board.is_check():
                score += 5

            board.pop()

            # updating best move
            if score > best_score:
                best_score, best_move = score, move

        if best_move:
            board.push(best_move) # temporarily making the best move
            self.seen_positions.append(board.fen().split(' ')[0]) # adding to memory
            if len(self.seen_positions) > self.seen_positions_window: # stopping if window is full
                self.seen_positions.pop(0)
            board.pop()

        return best_move.uci() if best_move else None


    def _critical_move(self, fen: str) -> Optional[str]:
        """
        Function with rules to check before the model.
        """
        board = chess.Board(fen)

        # checkmate
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move.uci()
            board.pop()

        # pawn to queen: when reaching last rank of the board
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square) # the piece that is being moved
            if piece and piece.piece_type == chess.PAWN: # is it a pawn?
                rank = chess.square_rank(move.to_square) # rank of destination square
                if (piece.color == chess.WHITE and rank == 7) or \
                   (piece.color == chess.BLACK and rank == 0): # checking if it reaches the last rank (then a pawn can become a queen)
                    promotion_move = chess.Move(
                        move.from_square, move.to_square, promotion=chess.QUEEN # we want it to become a queen
                    )
                    if promotion_move in board.legal_moves: # legal yes or no?
                        return promotion_move.uci()

        # when opponent has no pieces: use endgame function
        if self._endgame(board):
            return self._endgame_move(board)

        return None


    def _heuristic_move(self, fen: str) -> Optional[str]:
        """
        The safety net: only used if the model fails.
        """
        board = chess.Board(fen)

        # 1. best capture
        best_capture = None
        best_value = 0
        for move in board.legal_moves:
            if board.is_capture(move): # checking if the move is a capture
                captured = board.piece_at(move.to_square)
                # if captured piece has a higher value: updating best value
                if captured and self.piece_values.get(captured.piece_type, 0) > best_value:
                    if self._safe_move(board, move): # only if safe
                        best_value = self.piece_values[captured.piece_type]
                        best_capture = move # updating best capture
        if best_capture:
            return best_capture.uci()

        # 2. castling (to protect the king, but is most relevant at beginning of game)
        if board.fullmove_number <= 20: # first 20 moves
            for move in board.legal_moves:
                if board.is_castling(move):
                    return move.uci()

        # 3. give check
        for move in board.legal_moves:
            board.push(move)
            if board.is_check():
                board.pop()
                return move.uci()
            board.pop()

        # 4. random legal move
        moves = list(board.legal_moves) # all legal moves
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
                            if self._safe_move(board, move):
                                return move_str
                            # Move is legal but unsafe: it tries again (max 3 tries)
                    except ValueError:
                        pass

        except Exception:
            pass

        # Step 5: Model failed or kept suggesting unsafe moves: heuristics
        return self._heuristic_move(fen)
