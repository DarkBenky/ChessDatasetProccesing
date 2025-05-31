import pygame
import numpy as np
import chess
from chess import engine
import process
import tensorflow as tf
import os
import pickle
import pandas as pd
import time
from tensorflow.keras.callbacks import TensorBoard
import tensorboard
import datetime
from functools import lru_cache
import threading
import queue

MOVE_LIMIT = 125
GAMES_PER_TRAINING = 5  # Train after every 10 games
LOOK_BACK = MOVE_LIMIT * GAMES_PER_TRAINING  # Look back at last N moves for training
PARALLEL_GAMES = 2

# Chess piece image filenames
BISHOP_BLACK = "chessImages/BishupBlack.png"
KING_BLACK = "chessImages/kingBlack.png"
KNIGHT_BLACK = "chessImages/KnightBlack.png"
PAWN_BLACK = "chessImages/PawnBlack.png"
QUEEN_BLACK = "chessImages/QueenBlack.png"
ROOK_BLACK = "chessImages/RookBlack.png"

BISHOP_WHITE = "chessImages/BishupWhite.png"
KING_WHITE = "chessImages/KinkWhite.png"
KNIGHT_WHITE = "chessImages/KnightWhite.png"
PAWN_WHITE = "chessImages/PawnWhite.png"
QUEEN_WHITE = "chessImages/QueeenWhite.png"
ROOK_WHITE = "chessImages/RookWhite.png"


class ChessGame:
    def __init__(self,
                 model,
                 engine_path,
                 play_against_engine=False,
                 parallel_games_enabled=False,
                 parallel_games=1,
                 visual=True):
        self.model = model
        self.engine_path = engine_path  # Store the engine path
        self.engine = engine.SimpleEngine.popen_uci(engine_path)
        self.board = chess.Board()
        self.game_over = False
        self.play_against_engine = play_against_engine
        self.parallel_games_enabled = parallel_games_enabled
        self.parallel_games = parallel_games
        self.visual = visual
        # data shaped as board state, move history, and game status and eval from engine
        self.boards = []
        self.moves = []
        self.eval = []
        self.status = []
        # Remove ELO tracking
        # self.bestElo = 500  # Default ELO rating
        # Load move dictionaries
        with open('move_to_number.pkl', 'rb') as f:
            self.move_to_number = pickle.load(f)
        with open('number_to_move.pkl', 'rb') as f:
            self.number_to_move = pickle.load(f)

        print(f"Loaded move dictionaries with {len(self.move_to_number)} moves.")
        print(f"Loaded model with input shape: {self.model.input_shape} and output shape: {self.model.output_shape}")

        
        # Initialize TensorBoard logging
        log_dir = f"logs/chess_training_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        )
        
        # Create summary writer for custom metrics
        self.train_summary_writer = tf.summary.create_file_writer(log_dir + '/train')
        
        # pygame only for visual instance
        if self.visual:
            # Initialize pygame
            pygame.init()
            self.BOARD_SIZE = 640
            self.SQUARE_SIZE = self.BOARD_SIZE // 8
            self.screen = pygame.display.set_mode((self.BOARD_SIZE, self.BOARD_SIZE + 100))
            pygame.display.set_caption("Chess AI Self-Play")
            self.clock = pygame.time.Clock()
            
            # Colors
            self.WHITE = (240, 217, 181)
            self.BLACK = (181, 136, 99)
            self.TEXT_COLOR = (0, 0, 0)
            
            # Font for text
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
            # load piece images
            self.piece_images = {
                'b': pygame.image.load(BISHOP_BLACK),
                'k': pygame.image.load(KING_BLACK),
                'n': pygame.image.load(KNIGHT_BLACK),
                'p': pygame.image.load(PAWN_BLACK),
                'q': pygame.image.load(QUEEN_BLACK),
                'r': pygame.image.load(ROOK_BLACK),
                'B': pygame.image.load(BISHOP_WHITE),
                'K': pygame.image.load(KING_WHITE),
                'N': pygame.image.load(KNIGHT_WHITE),
                'P': pygame.image.load(PAWN_WHITE),
                'Q': pygame.image.load(QUEEN_WHITE),
                'R': pygame.image.load(ROOK_WHITE)
            }
        else:
            self.BOARD_SIZE = self.SQUARE_SIZE = None
            self.screen = self.clock = None
            
        self.game_count = 0
        self.games_per_training = GAMES_PER_TRAINING  # Train after every 5 games
        self.training_step = 0  # Track training steps for TensorBoard
        self.checkmates_found = 0  # Track checkmates found in current game
        print("ChessGame initialized with model and engine.")
        
        # spawn headless games if requested (only in main visual)
        if self.visual and self.parallel_games_enabled and self.parallel_games > 1:
            for _ in range(self.parallel_games - 1):
                g = ChessGame(model,
                              engine_path,
                              play_against_engine,
                              parallel_games_enabled=False,
                              parallel_games=1,
                              visual=False)
                threading.Thread(target=g.headless_run, daemon=True).start()

        # Add training queue and thread management
        self.training_queue = queue.Queue()
        self.training_thread = None
        self.is_training = False
        self.training_complete = threading.Event()

    def resetGame(self):
        self.board = chess.Board()
        self.game_over = False
        self.boards = []
        self.moves = []
        self.eval = []
        self.status = []
        self.checkmates_found = 0  # Reset checkmate counter
        
        # load random starting position from high_rated_games.csv
        try:
            df = pd.read_csv('high_rated_games.csv')
            random_index = np.random.randint(len(df))
            # choose a random game from the between 1. and 10. move
            random_length = np.random.randint(1, 10)
            moves_str = df.iloc[random_index]['moves']
            moves_list = moves_str.split(' ')
            
            for i in range(min(random_length, len(moves_list))):
                move = moves_list[i]
                try:
                    self.board.push(chess.Move.from_uci(move))
                except:
                    break
        except:
            # If CSV not found or error, start from standard position
            pass

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = self.WHITE if (row + col) % 2 == 0 else self.BLACK
                pygame.draw.rect(self.screen, color, 
                               (col * self.SQUARE_SIZE, row * self.SQUARE_SIZE, 
                                self.SQUARE_SIZE, self.SQUARE_SIZE))
                
                # Draw piece symbols
                square = chess.square(col, 7 - row)
                piece = self.board.piece_at(square)
                if piece:
                    self.draw_piece(piece, col * self.SQUARE_SIZE, row * self.SQUARE_SIZE)

    def draw_piece(self, piece, x, y):
        # Simple text representation of pieces
        piece_symbols = {
            'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',
            'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚'
        }
        
        symbol = piece_symbols.get(piece.symbol(), piece.symbol())
        text = self.font.render(symbol, True, self.TEXT_COLOR)
        text_rect = text.get_rect(center=(x + self.SQUARE_SIZE // 2, y + self.SQUARE_SIZE // 2))
        self.screen.blit(text, text_rect)

  
        # Draw piece images instead of text
        if piece.color == chess.WHITE:
            piece_image = self.piece_images[piece.symbol().upper()]
        else:
            piece_image = self.piece_images[piece.symbol().lower()]
        piece_image = pygame.transform.scale(piece_image, (self.SQUARE_SIZE, self.SQUARE_SIZE))
        self.screen.blit(piece_image, (x, y))

    def draw_info(self):
        # Draw game info below the board
        info_y = self.BOARD_SIZE + 10
        
        # Current turn with player info
        if self.play_against_engine:
            current_player = "Model (White)" if self.board.turn else "Engine (Black)"
            if not self.board.turn:  # It's Black's turn, but we show who just moved
                current_player = "Engine (Black)" if len(self.moves) % 2 == 1 else "Model (White)"
            elif self.board.turn:  # It's White's turn
                current_player = "Model (White)" if len(self.moves) % 2 == 0 else "Engine (Black)"
        else:
            current_player = f"Model ({'White' if self.board.turn else 'Black'})"
        
        turn_text = f"Turn: {current_player}"
        turn_surface = self.small_font.render(turn_text, True, self.TEXT_COLOR)
        self.screen.blit(turn_surface, (10, info_y))
        
        # Move count
        move_text = f"Moves: {len(self.moves)}"
        move_surface = self.small_font.render(move_text, True, self.TEXT_COLOR)
        self.screen.blit(move_surface, (150, info_y))
        
        # Last evaluation
        if self.eval:
            eval_text = f"Eval: {self.eval[-1] if self.eval[-1] is not None else 'N/A'}"
            eval_surface = self.small_font.render(eval_text, True, self.TEXT_COLOR)
            self.screen.blit(eval_surface, (250, info_y))
        
        # Game status
        if self.board.is_game_over():
            if self.board.is_checkmate():
                winner = "Black" if self.board.turn else "White"
                if self.play_against_engine:
                    if winner == "White":
                        status_text = "Checkmate! Model (White) wins!"
                    else:
                        status_text = "Checkmate! Engine (Black) wins!"
                else:
                    status_text = f"Checkmate! {winner} wins!"
            elif self.board.is_stalemate():
                status_text = "Stalemate! Draw!"
            else:
                status_text = "Game Over! Draw!"
            
            status_surface = self.font.render(status_text, True, (255, 0, 0))
            self.screen.blit(status_surface, (10, info_y + 30))

    def get_piece_value(self, piece_type):
        """Return the standard point value of a chess piece"""
        if piece_type == chess.PAWN:
            return 1
        elif piece_type in (chess.KNIGHT, chess.BISHOP):
            return 3
        elif piece_type == chess.ROOK:
            return 5
        elif piece_type == chess.QUEEN:
            return 9
        elif piece_type == chess.KING:
            return 100  # King has effectively infinite value
        return 0

    def get_model_move(self):
        # get encoded board state to model format
        board_state = process.board_to_array(self.board)
        
        # predict the next move using the model
        input_data = np.array([board_state])
        predictions = self.model.predict(input_data, verbose=0)
        
        # Get current evaluation to determine winning status
        current_eval = self.get_engine_evaluation() if not self.eval else (self.eval[-1] if self.eval[-1] is not None else 0)
        
        # Determine if we're winning and by how much
        is_winning = False
        winning_margin = 0
        aggression_multiplier = 1.0
        
        if current_eval is not None:
            # From current player's perspective
            player_eval = current_eval if self.board.turn else -current_eval
            
            if player_eval > 100:  # Winning by more than 1 pawn
                is_winning = True
                winning_margin = player_eval
                
                if player_eval > 500:  # Major advantage (5+ pawns)
                    aggression_multiplier = 2.0
                    print(f"Major advantage detected ({player_eval}): Maximum aggression mode")
                elif player_eval > 300:  # Significant advantage (3+ pawns)
                    aggression_multiplier = 1.6
                    print(f"Significant advantage detected ({player_eval}): High aggression mode")
                elif player_eval > 150:  # Moderate advantage (1.5+ pawns)
                    aggression_multiplier = 1.3
                    print(f"Moderate advantage detected ({player_eval}): Increased aggression mode")
        
        # Get all legal moves first
        legal_moves = list(self.board.legal_moves)
        legal_moves_in_dict = []
        
        # Filter legal moves that are in dictionary using algebraic notation
        for move in legal_moves:
            move_san = self.board.san(move)  # Convert to algebraic notation
            if move_san in self.move_to_number:
                move_index = self.move_to_number[move_san]
                score = predictions[0][move_index]
                
                # Calculate capture bonus with winning aggression
                capture_bonus = 0
                if self.board.is_capture(move):
                    # Get the value of the captured piece
                    to_square = move.to_square
                    captured_piece = self.board.piece_at(to_square)
                    if captured_piece:
                        captured_value = self.get_piece_value(captured_piece.piece_type)
                        
                        # Get the value of the capturing piece
                        from_square = move.from_square
                        capturing_piece = self.board.piece_at(from_square)
                        if capturing_piece:
                            capturing_value = self.get_piece_value(capturing_piece.piece_type)
                            
                            # Calculate exchange value - higher is better
                            exchange_value = captured_value - capturing_value
                            
                            # Add slight randomness factor (0.8-1.0) to vary capture eagerness
                            randomness = 0.8 + 0.2 * np.random.random()
                            
                            # Apply aggression multiplier to capture bonuses
                            base_capture_bonus = 0
                            if exchange_value >= 5:  # Extremely good capture (e.g., pawn takes queen)
                                base_capture_bonus = min(0.4, 0.25 + exchange_value * 0.02) * randomness
                            elif exchange_value > 1:  # Good capture (captured piece worth more)
                                base_capture_bonus = min(0.2, exchange_value * 0.03) * randomness
                            elif exchange_value == 0:  # Equal exchange
                                base_capture_bonus = 0.03 * randomness
                            elif exchange_value >= -2:  # Slightly bad capture but potentially tactical
                                base_capture_bonus = 0.01 * randomness
                            else:  # Very unfavorable - discourage most of the time
                                base_capture_bonus = (0.01 - 0.02 * abs(exchange_value)) * randomness
                            
                            # Apply aggression multiplier - more aggressive when winning
                            capture_bonus = base_capture_bonus * aggression_multiplier
                            
                            if is_winning and base_capture_bonus > 0:
                                print(f"Aggressive capture: {move_san}, base: {base_capture_bonus:.3f}, "
                                      f"multiplier: {aggression_multiplier:.1f}, final: {capture_bonus:.3f}")
                
                # Check bonus - heavily boosted when winning
                check_bonus = 0
                # Make a copy of the board to test the move
                test_board = self.board.copy()
                test_board.push(move)
                
                if test_board.is_check():
                    base_check_bonus = 0.08  # Base check bonus
                    
                    if is_winning:
                        # Much higher check bonus when winning
                        check_bonus = base_check_bonus * aggression_multiplier * 1.5  # Extra boost for checks
                        print(f"Aggressive check: {move_san}, bonus: {check_bonus:.3f}")
                    else:
                        check_bonus = base_check_bonus
                    
                    # Extra bonus for checkmate
                    if test_board.is_checkmate():
                        checkmate_bonus = 1.0  # Massive bonus for checkmate
                        check_bonus += checkmate_bonus
                        print(f"CHECKMATE FOUND: {move_san}, total bonus: {check_bonus:.3f}")
                
                # Attack bonus - encourage attacking moves when winning
                attack_bonus = 0
                if is_winning:
                    piece_at_from = self.board.piece_at(move.from_square)
                    if piece_at_from:
                        # Bonus for attacking opponent's pieces
                        attacked_squares = []
                        test_board_attack = self.board.copy()
                        test_board_attack.push(move)
                        
                        # Check if this move attacks any opponent pieces
                        piece_attacks = test_board_attack.attacks(move.to_square)
                        for square in piece_attacks:
                            target_piece = test_board_attack.piece_at(square)
                            if target_piece and target_piece.color != self.board.turn:
                                attack_value = self.get_piece_value(target_piece.piece_type)
                                attack_bonus += 0.02 * attack_value * aggression_multiplier
                        
                        if attack_bonus > 0:
                            print(f"Attack bonus: {move_san}, bonus: {attack_bonus:.3f}")
                
                # Reduce castling and defensive bonuses when winning significantly
                castling_bonus = 0
                if self.board.is_castling(move):
                    move_count = len(self.moves)
                    
                    if move_count < 25:  # Early game
                        base_castling_bonus = 0.15 + 0.05 * np.random.random()
                    elif move_count < 40:  # Mid game
                        base_castling_bonus = 0.08 + 0.04 * np.random.random()
                    else:  # Late game
                        base_castling_bonus = 0.03 + 0.02 * np.random.random()
                    
                    # Reduce castling bonus when winning (focus on attack)
                    if is_winning and winning_margin > 300:
                        castling_bonus = base_castling_bonus * 0.5  # Reduce by half
                        print(f"Reduced castling bonus due to winning position: {castling_bonus:.3f}")
                    else:
                        castling_bonus = base_castling_bonus
                    
                    # Additional bonus for kingside castling
                    if move.to_square in [chess.G1, chess.G8]:
                        castling_bonus *= 1.2
                
                # Early king penalty (keep as is)
                early_king_penalty = 0
                if self.board.piece_at(move.from_square) and self.board.piece_at(move.from_square).piece_type == chess.KING:
                    if not self.board.is_castling(move):
                        move_count = len(self.moves)
                        
                        current_player = self.board.turn
                        can_still_castle = (
                            (current_player == chess.WHITE and (self.board.has_kingside_castling_rights(chess.WHITE) or self.board.has_queenside_castling_rights(chess.WHITE))) or
                            (current_player == chess.BLACK and (self.board.has_kingside_castling_rights(chess.BLACK) or self.board.has_queenside_castling_rights(chess.BLACK)))
                        )
                        
                        if move_count < 25 and can_still_castle:
                            early_king_penalty = -0.25 - 0.05 * np.random.random()
                        elif move_count < 35 and can_still_castle:
                            early_king_penalty = -0.15 - 0.05 * np.random.random()
                        elif move_count < 45 and can_still_castle:
                            early_king_penalty = -0.08 - 0.02 * np.random.random()
                
                # Promotion bonus (boosted when winning)
                promotion_bonus = 0
                if move.promotion:
                    move_count = len(self.moves)
                    
                    if move_count < 30:
                        base_bonus = 0.20
                    elif move_count < 50:
                        base_bonus = 0.15
                    else:
                        base_bonus = 0.25
                    
                    piece_multipliers = {
                        chess.QUEEN: 1.0,
                        chess.ROOK: 0.7,
                        chess.BISHOP: 0.4,
                        chess.KNIGHT: 0.4
                    }
                    
                    piece_multiplier = piece_multipliers.get(move.promotion, 0.2)
                    randomness = 0.9 + 0.2 * np.random.random()
                    
                    # Boost promotion bonus when winning
                    final_multiplier = aggression_multiplier if is_winning else 1.0
                    promotion_bonus = base_bonus * piece_multiplier * randomness * final_multiplier
                    
                    piece_names = {
                        chess.QUEEN: "Queen",
                        chess.ROOK: "Rook",
                        chess.BISHOP: "Bishop",
                        chess.KNIGHT: "Knight"
                    }
                    piece_name = piece_names.get(move.promotion, "Unknown")
                    if is_winning:
                        print(f"Aggressive promotion to {piece_name}: {move_san}, bonus: {promotion_bonus:.3f}")
                
                # Endgame pawn bonus (keep existing logic)
                endgame_pawn_bonus = 0
                if self.board.piece_at(move.from_square) and self.board.piece_at(move.from_square).piece_type == chess.PAWN:
                    total_material = 0
                    for square in chess.SQUARES:
                        piece = self.board.piece_at(square)
                        if piece and piece.piece_type not in [chess.KING, chess.PAWN]:
                            total_material += self.get_piece_value(piece.piece_type)
                    
                    if total_material <= 20:
                        from_rank = chess.square_rank(move.from_square)
                        to_rank = chess.square_rank(move.to_square)
                        
                        if self.board.turn:
                            rank_advance = to_rank - from_rank
                            advancement = to_rank
                        else:
                            rank_advance = from_rank - to_rank
                            advancement = 7 - to_rank
                        
                        if rank_advance > 0:
                            if total_material <= 10:
                                base_endgame_bonus = 0.12
                            elif total_material <= 15:
                                base_endgame_bonus = 0.08
                            else:
                                base_endgame_bonus = 0.05
                            
                            advancement_multiplier = 1.0 + (advancement * 0.3)
                            
                            if rank_advance == 2:
                                advancement_multiplier *= 1.4
                            
                            randomness = 0.9 + 0.2 * np.random.random()
                            endgame_pawn_bonus = base_endgame_bonus * advancement_multiplier * randomness
                
                # Add all bonuses and penalties to the prediction score
                adjusted_score = score + capture_bonus + check_bonus + attack_bonus + castling_bonus + early_king_penalty + promotion_bonus + endgame_pawn_bonus
                legal_moves_in_dict.append((move, move_index, adjusted_score))
        
        # If we have legal moves in dictionary
        if legal_moves_in_dict:
            # Sort by adjusted prediction probability (highest first)
            legal_moves_in_dict.sort(key=lambda x: x[2], reverse=True)
            
            # More focused move selection when winning
            if is_winning and winning_margin > 200:
                # When winning significantly, be more decisive
                top_k = min(2, len(legal_moves_in_dict))  # Very focused selection
                chosen_move = legal_moves_in_dict[np.random.randint(top_k)][0]
                print(f"Winning position: Selecting from top {top_k} moves")
            elif self.eval and abs(self.eval[-1] or 0) > 200:
                top_k = min(3, len(legal_moves_in_dict))
                chosen_move = legal_moves_in_dict[np.random.randint(top_k)][0]
            else:
                top_k = min(5, len(legal_moves_in_dict))
                chosen_move = legal_moves_in_dict[np.random.randint(top_k)][0]
            return chosen_move
        
        # Fallback: return a random legal move if no dictionary moves available
        if legal_moves:
            return legal_moves[np.random.randint(len(legal_moves))]
        
        return None

    def get_engine_move(self):
        """Get move from Stockfish engine with improved error handling"""
        try:
            # Remove the problematic debug call
            # self.engine.debug(False)  # This line causes the error
            
            # Send the current position to the engine explicitly
            result = self.engine.play(self.board, chess.engine.Limit(time=0.15))
            
            # Verify the move is legal before returning
            if result.move and result.move in self.board.legal_moves:
                return result.move
            else:
                print(f"Engine returned illegal move: {result.move}")
                # Fallback to random legal move
                legal_moves = list(self.board.legal_moves)
                return legal_moves[np.random.randint(len(legal_moves))] if legal_moves else None
                
        except chess.engine.EngineError as e:
            print(f"Engine error: {e}")
            # Try to recover by restarting the engine connection
            try:
                self.engine.quit()
                self.engine = engine.SimpleEngine.popen_uci(self.engine_path)
                print("Engine restarted successfully")
            except Exception as restart_error:
                print(f"Failed to restart engine: {restart_error}")
            
            # Fallback to random legal move
            legal_moves = list(self.board.legal_moves)
            return legal_moves[np.random.randint(len(legal_moves))] if legal_moves else None
            
        except Exception as e:
            print(f"Unexpected engine error: {e}")
            # Fallback to random legal move
            legal_moves = list(self.board.legal_moves)
            return legal_moves[np.random.randint(len(legal_moves))] if legal_moves else None

    def get_engine_evaluation(self, board=None):
        """Get evaluation from Stockfish engine with proper error handling"""
        if board is None:
            board = self.board
        
        try:
            # Create a fresh engine instance for evaluation to avoid state issues
            with engine.SimpleEngine.popen_uci(self.engine_path) as eval_engine:
                # Analyze position with timeout
                info = eval_engine.analyse(board, chess.engine.Limit(time=0.15))
                score = info['score'].relative
                
                # Handle mate scores
                if score.is_mate():
                    mate_in = score.mate()
                    if mate_in is None:
                        return 0  # Fallback for unknown mate
                    
                    # Convert mate to large numeric value
                    if mate_in > 0:
                        return 9999 - mate_in  # Closer mate = higher score
                    else:
                        return -9999 - mate_in  # Closer mate = lower score
                
                # Handle regular centipawn scores
                elif score.score() is not None:
                    cp_score = score.score()
                    return max(-9999, min(9999, cp_score))
                else:
                    return 0
                    
        except chess.engine.EngineTerminatedError:
            print("Engine terminated unexpectedly during evaluation")
            return 0
        except chess.engine.EngineError as e:
            print(f"Engine error during evaluation: {e}")
            return 0
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0

    def play_elo_test_game(self, test_engine, max_moves=60):
        """Play a single quick game for ELO testing with better error handling"""
        test_board = chess.Board()
        move_count = 0
        
        # Model plays white, engine plays black
        while not test_board.is_game_over() and move_count < max_moves:
            try:
                if test_board.turn:  # White (model) to move
                    # Get model move using existing logic
                    old_board = self.board
                    self.board = test_board
                    move = self.get_model_move()
                    self.board = old_board
                    
                    if move and move in test_board.legal_moves:
                        test_board.push(move)
                    else:
                        # Model failed to find legal move, it loses
                        return -1
                else:  # Black (engine) to move
                    try:
                        # Set a shorter time limit for test games
                        result = test_engine.play(test_board, chess.engine.Limit(time=0.1))
                        
                        # Verify the move is legal
                        if result.move and result.move in test_board.legal_moves:
                            test_board.push(result.move)
                        else:
                            print(f"Test engine returned illegal move: {result.move}")
                            # Engine failed, model wins
                            return 1
                    except chess.engine.EngineError as e:
                        print(f"Test engine error: {e}")
                        # Engine failed, model wins
                        return 1
                    except Exception as e:
                        print(f"Unexpected test engine error: {e}")
                        # Engine failed, model wins
                        return 1
                
                move_count += 1
                
            except Exception as e:
                print(f"Error in test game: {e}")
                # Call it a draw on unexpected errors
                return 0
        
        # Determine result
        if test_board.is_game_over():
            if test_board.is_checkmate():
                # Winner is opposite of current turn
                return 1 if not test_board.turn else -1
            else:
                return 0  # Draw
        else:
            # Game reached move limit, call it a draw
            return 0

    def estimate_elo_fast(self):
        """Fast ELO estimation with better error handling"""
        print("Starting fast ELO estimation...")
        start_time = time.time()
        
        results = []  # (opponent_elo, game_result)
        
        # Play one game against each level
        for level in STOCKFISH_LEVELS:
            opponent_elo = STOCKFISH_ELO_MAP[level]
            temp_engine = None
            
            try:
                # Create temporary engine with specific skill level
                temp_engine = engine.SimpleEngine.popen_uci(self.engine_path)
                # Remove the problematic debug call
                # temp_engine.debug(False)  # This line causes the error
                temp_engine.configure({"Skill Level": level})
                
                # Play a quick game
                result = self.play_elo_test_game(temp_engine, max_moves=120)
                results.append((opponent_elo, result))
                
                print(f"vs Stockfish Level {level} (ELO ~{opponent_elo}): {'Win' if result == 1 else 'Loss' if result == -1 else 'Draw'}")
                
            except Exception as e:
                print(f"Error testing against level {level}: {e}")
                results.append((opponent_elo, 0))  # Assume draw on error
            finally:
                if temp_engine:
                    try:
                        temp_engine.quit()
                    except:
                        pass

    def calculate_performance_elo(self, results):
        """Calculate performance ELO from test results"""
        if not results:
            return 1200  # Default rating
        
        total_score = 0
        total_weighted_elo = 0
        total_weight = 0
        
        for opponent_elo, result in results:
            # Convert result to score (1 = win, 0.5 = draw, 0 = loss)
            score = (result + 1) / 2
            total_score += score
            
            # Weight by opponent strength (higher rated opponents matter more)
            weight = 1.0 + (opponent_elo - 1500) / 1000
            weight = max(0.5, weight)  # Minimum weight
            
            total_weighted_elo += opponent_elo * weight
            total_weight += weight
        
        avg_opponent_elo = total_weighted_elo / total_weight if total_weight > 0 else 1500
        score_percentage = total_score / len(results)
        
        # Performance rating formula
        if score_percentage == 1.0:
            performance_elo = avg_opponent_elo + 400  # Assume 400 points stronger
        elif score_percentage == 0.0:
            performance_elo = avg_opponent_elo - 400  # Assume 400 points weaker
        else:
            # Use inverse normal distribution approximation
            # This is a simplified version of the FIDE formula
            if score_percentage > 0.99:
                score_percentage = 0.99
            elif score_percentage < 0.01:
                score_percentage = 0.01
            
            # Approximate expected score difference to ELO difference
            # E(s) = 1 / (1 + 10^((opponent_elo - player_elo) / 400))
            # Solving for player_elo given E(s) = score_percentage
            elo_diff = -400 * np.log10((1 / score_percentage) - 1)
            performance_elo = avg_opponent_elo + elo_diff
        
        # Clamp to reasonable range
        performance_elo = max(800, min(3000, performance_elo))
        
        return performance_elo
    
    def train_model(self):
        """Start training in a separate thread to avoid blocking the main thread"""
        if not self.is_training:
            # Set is_training immediately to prevent multiple training threads
            self.is_training = True
            self.training_thread = threading.Thread(target=self.train_model_threaded, daemon=True)
            self.training_thread.start()
            print("Training started in background thread...")
        else:
            print("Training already in progress, skipping...")

    def train_model_threaded(self):
        """Thread-safe training function that runs in background"""
        # Don't set is_training here since it's already set in train_model()
        self.training_complete.clear()
        
        try:
            print("Starting model training in background thread...")
            df = pd.read_csv('game_data.csv')
            latest_games = df.tail(self.games_per_training * LOOK_BACK * PARALLEL_GAMES)

            # Add periodic yield to prevent blocking
            import time
            
            # locals for speed
            move_to_number = self.move_to_number
            n_moves = len(move_to_number)
            boards, move_targets = [], []

            # initialize counters
            total_blunders = 0
            total_mates_found = 0
            moves_not_in_dict = 0
            total_moves_processed = 0

            print(f"Processing {len(latest_games)} total game records...")
            print(f"Number of unique games: {len(latest_games.groupby('game_id'))}")

            game_count = 0
            for game_id, group in latest_games.groupby('game_id'):
                game_count += 1
                
                # Yield control every 10 games to keep UI responsive
                if game_count % 10 == 0:
                    time.sleep(0.001)  # Very brief yield
                
                # per‐game counters
                game_blunders = 0
                game_mates = 0

                game_data = group.reset_index(drop=True)
                temp_board = chess.Board()

                for i, row in game_data.iterrows():
                    total_moves_processed += 1
                    board = np.array(eval(row['boards']))
                    move_str = row['moves']

                    # Convert move to SAN format if needed
                    try:
                        if move_str in move_to_number:
                            final_move_str = move_str
                        else:
                            try:
                                uci_move = chess.Move.from_uci(move_str)
                                if uci_move in temp_board.legal_moves:
                                    final_move_str = temp_board.san(uci_move)
                                else:
                                    moves_not_in_dict += 1
                                    continue
                            except:
                                moves_not_in_dict += 1
                                continue
                        
                        try:
                            if move_str in move_to_number:
                                temp_board.push_san(move_str)
                            else:
                                temp_board.push(chess.Move.from_uci(move_str))
                        except:
                            break
                            
                    except Exception as e:
                        print(f"Error processing move {move_str}: {e}")
                        moves_not_in_dict += 1
                        continue

                    eval_score = row['eval']
                    if pd.isna(eval_score) or eval_score is None or not isinstance(eval_score, (int, float)):
                        eval_score = 0
                    else:
                        try:
                            eval_score = float(eval_score)
                            if np.isnan(eval_score) or np.isinf(eval_score):
                                eval_score = 0
                        except (ValueError, TypeError):
                            eval_score = 0
                    
                    move_quality = 0.25
                    
                    if i > 1:
                        prev_eval = game_data.iloc[i-2]['eval']
                        if pd.notna(prev_eval) and isinstance(prev_eval, (int, float)):
                            try:
                                prev_eval = float(prev_eval)
                                if not np.isnan(prev_eval) and not np.isinf(prev_eval):
                                    eval_change = eval_score - prev_eval
                                    # current eval -   previous eval = eval change
                                    # 0            -   - 500         = 500
                                    # 500          -   0             = 500
                                    # -500         -   0             = -500
                                    # 0            -   500           = -500
                                    
                                    if eval_change < -125:
                                        # Scale penalty based on how bad the blunder was
                                        penalty_multiplier = min(3.0, abs(eval_change) / 400.0)  # Cap at 3x penalty
                                        base_penalty = 0.4
                                        total_penalty = base_penalty * penalty_multiplier
                                        
                                        move_quality = max(0.001, move_quality - total_penalty)
                                        print(f"Blunder detected: eval change {eval_change}, penalty: -{total_penalty:.2f}, quality: {move_quality:.2f}")
                                    else:
                                        move_quality = move_quality + (eval_change / 1000.0)
                                        move_quality = max(0.001, min(0.9, move_quality))
                            except (ValueError, TypeError):
                                pass

                    if final_move_str in move_to_number:
                        played_move_idx = move_to_number[final_move_str]
                        
                        target_dist = np.full(n_moves, 1e-6)
                        target_dist[played_move_idx] = max(0.1, move_quality)
                        
                        try:
                            temp_board_copy = chess.Board(temp_board.fen())
                            temp_board_copy.pop()
                            legal_moves_here = list(temp_board_copy.legal_moves)
                            
                            for legal_move in legal_moves_here[:min(5, len(legal_moves_here))]:
                                try:
                                    legal_san = temp_board_copy.san(legal_move)
                                    if legal_san in move_to_number and legal_san != final_move_str:
                                        legal_idx = move_to_number[legal_san]
                                        target_dist[legal_idx] = np.random.uniform(0.001, 0.01)
                                except:
                                    continue
                        except:
                            pass
                        
                        target_dist = target_dist / np.sum(target_dist)
                        
                        boards.append(board)
                        move_targets.append(target_dist)
                    else:
                        moves_not_in_dict += 1
                        continue

            print(f"Total moves processed: {total_moves_processed}")
            print(f"Total moves not in dictionary: {moves_not_in_dict}")
            print(f"Training positions created: {len(boards)}")
            print(f"Dictionary coverage: {((total_moves_processed - moves_not_in_dict) / total_moves_processed * 100):.1f}%")
            print(f"Total checkmates found during training: {total_mates_found}")
            print(f"Total blunders found during training: {total_blunders}")
            
            if len(boards) > 0:
                X = np.array(boards)
                y = np.array(move_targets)
                
                print(f"Training on {len(boards)} positions...")
                
                # Configure model in the training thread
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy', 'top_k_categorical_accuracy']
                )
                
                # Use a custom callback to periodically yield control
                class ResponsiveCallback(tf.keras.callbacks.Callback):
                    def on_batch_end(self, batch, logs=None):
                        if batch % 50 == 0:  # Yield every 50 batches
                            time.sleep(0.001)
                
                responsive_callback = ResponsiveCallback()
                
                history = self.model.fit(
                    X, y, 
                    epochs=3,
                    batch_size=64,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=[self.tensorboard_callback, responsive_callback]
                )
                
                val_loss = history.history.get('val_loss', [])
                if val_loss and len(val_loss) > 1:
                    if val_loss[-1] > val_loss[0]:
                        print("WARNING: Validation loss increased - consider reducing learning rate further")
                    else:
                        print(f"✅ Validation loss improved: {val_loss[0]:.4f} → {val_loss[-1]:.4f}")
                
                avg_move_quality = np.mean([np.max(target) for target in move_targets])
                self.log_custom_metrics_updated(avg_move_quality, latest_games, total_mates_found, total_blunders)
                
                # Remove ELO estimation
                # try:
                #     estimated_elo = self.estimate_elo_fast()
                #     if estimated_elo > self.bestElo:
                #         self.bestElo = estimated_elo
                #         print(f"New best ELO estimated: {self.bestElo:.0f}")
                #     else:
                #         print(f"ELO estimation did not improve: {estimated_elo:.0f} (best: {self.bestElo:.0f})")
                # except Exception as e:
                #     print(f"ELO estimation failed: {e}")
                
                self.model.save('chess_transformer_model.keras')
                print(f"Model trained on {len(boards)} positions and saved.")
                self.training_step += 1
            else:
                print("No valid training data found. Check move dictionary compatibility.")
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_training = False
            self.training_complete.set()
            print("Training thread completed.")

    def train_model(self):
        """Start training in a separate thread to avoid blocking the main thread"""
        if not self.is_training:
            # Set is_training immediately to prevent multiple training threads
            self.is_training = True
            self.training_thread = threading.Thread(target=self.train_model_threaded, daemon=True)
            self.training_thread.start()
            print("Training started in background thread...")
        else:
            print("Training already in progress, skipping...")

    def log_custom_metrics_updated(self, avg_move_quality, latest_games, total_mates_found, total_blunders):
        """Updated logging function with checkmate and blunder statistics"""
        with self.train_summary_writer.as_default():
            tf.summary.scalar('training/average_move_quality', avg_move_quality, step=self.training_step)
            
            # Log game statistics
            wins = sum(1 for _, row in latest_games.iterrows() if row['status'] == 1)
            losses = sum(1 for _, row in latest_games.iterrows() if row['status'] == -1)
            draws = sum(1 for _, row in latest_games.iterrows() if row['status'] == 0)
            total_games = len(latest_games.groupby('game_id'))
            
            if total_games > 0:
                tf.summary.scalar('games/win_rate', wins / total_games, step=self.training_step)
                tf.summary.scalar('games/loss_rate', losses / total_games, step=self.training_step)
                tf.summary.scalar('games/draw_rate', draws / total_games, step=self.training_step)
                tf.summary.scalar('games/total_games', total_games, step=self.training_step)
                
                # Log checkmate statistics
                tf.summary.scalar('checkmates/total_mates_found', total_mates_found, step=self.training_step)
                tf.summary.scalar('checkmates/mates_per_game', total_mates_found / total_games, step=self.training_step)
                
                games_with_mates = sum(1 for game_id, group in latest_games.groupby('game_id') 
                                     if any(abs(row['eval']) > 5000 for _, row in group.iterrows() 
                                           if pd.notna(row['eval']) and isinstance(row['eval'], (int, float)) and not np.isnan(row['eval'])))
                tf.summary.scalar('checkmates/games_with_mates', games_with_mates, step=self.training_step)
                
                # Log blunder statistics
                tf.summary.scalar('blunders/total_blunders_found', total_blunders, step=self.training_step)
                tf.summary.scalar('blunders/blunders_per_game', total_blunders / total_games, step=self.training_step)
                
                games_with_blunders = 0
                for game_id, group in latest_games.groupby('game_id'):
                    game_evals = []
                    for _, row in group.iterrows():
                        eval_val = row['eval']
                        if pd.notna(eval_val) and isinstance(eval_val, (int, float)):
                            try:
                                eval_val = float(eval_val)
                                if not np.isnan(eval_val) and not np.isinf(eval_val):
                                    game_evals.append(eval_val)
                                else:
                                    game_evals.append(0)
                            except (ValueError, TypeError):
                                game_evals.append(0)
                    
                    for j in range(1, len(game_evals)):
                        eval_change = game_evals[j] - game_evals[j-1]
                        if eval_change < -200:  # Blunder detected
                            games_with_blunders += 1
                            break
                
                tf.summary.scalar('blunders/games_with_blunders', games_with_blunders, step=self.training_step)
                tf.summary.scalar('blunders/blunder_rate', total_blunders / len(latest_games) if len(latest_games) > 0 else 0, step=self.training_step)
            
            # Log evaluation statistics with proper NaN handling
            valid_evals = []
            for _, row in latest_games.iterrows():
                eval_val = row['eval']
                if pd.notna(eval_val) and isinstance(eval_val, (int, float)):
                    try:
                        eval_val = float(eval_val)
                        if not np.isnan(eval_val) and not np.isinf(eval_val):
                            valid_evals.append(eval_val)
                    except (ValueError, TypeError):
                        pass
            
            if valid_evals:
                tf.summary.scalar('evaluation/average_eval', np.mean(valid_evals), step=self.training_step)
                tf.summary.scalar('evaluation/eval_std', np.std(valid_evals), step=self.training_step)
                tf.summary.scalar('evaluation/valid_eval_count', len(valid_evals), step=self.training_step)
                tf.summary.scalar('evaluation/total_positions', len(latest_games), step=self.training_step)
            
            # Log move count statistics
            move_counts = [len(group) for _, group in latest_games.groupby('game_id')]
            if move_counts:
                tf.summary.scalar('games/average_moves', np.mean(move_counts), step=self.training_step)
                tf.summary.scalar('games/max_moves', np.max(move_counts), step=self.training_step)
                tf.summary.scalar('games/min_moves', np.min(move_counts), step=self.training_step)
        
        self.train_summary_writer.flush()

    def play(self):
        # headless shortcut
        if not self.visual:
            return self.headless_run()
        
        running = True
        auto_play = True
        move_delay = 0  # milliseconds between moves
        last_move_time = 0
        
        while running:
            current_time = pygame.time.get_ticks()
            
            # Handle events more frequently
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        auto_play = not auto_play
                    elif event.key == pygame.K_r:
                        self.resetGame()
                        last_move_time = current_time
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_t:  # Add 'T' key to check training status
                        if self.is_training:
                            print("Training in progress...")
                        else:
                            print("No training in progress.")
            
            # Auto-play logic
            if auto_play and not self.game_over and current_time - last_move_time > move_delay:
                self.make_move()
                last_move_time = current_time
                
                if self.game_over:
                    self.save_game_data()
                    self.game_count += 1
                    
                    # Start training in background thread (non-blocking)
                    if self.game_count % self.games_per_training == 0:
                        self.train_model()
                    
                    self.resetGame()
                    last_move_time = current_time
            
            # Draw everything
            self.screen.fill((255, 255, 255))
            self.draw_board()
            self.draw_info()
            
            # Show training status with more detail
            if self.is_training:
                training_text = self.small_font.render("🔄 Training in progress...", True, (255, 0, 0))
                self.screen.blit(training_text, (10, self.BOARD_SIZE + 70))
                # Show that the game is still responsive
                responsive_text = self.small_font.render("(Game continues in background)", True, (100, 100, 100))
                self.screen.blit(responsive_text, (200, self.BOARD_SIZE + 70))
            
            # Draw controls
            controls = ["SPACE: Pause/Resume", "R: Reset", "T: Training Status", "ESC: Quit"]
            for i, control in enumerate(controls):
                text = self.small_font.render(control, True, self.TEXT_COLOR)
                self.screen.blit(text, (350, self.BOARD_SIZE + 10 + i * 15))
            
            pygame.display.flip()
            self.clock.tick(60)  # Keep responsive at 60 FPS
        
        # Cleanup
        if self.training_thread and self.training_thread.is_alive():
            print("Waiting for training to complete...")
            self.training_complete.wait(timeout=30)
        
        self.engine.quit()
        pygame.quit()

    def headless_run(self):
        """Run without graphics."""
        self.resetGame()
        while not self.game_over:
            self.make_move()
        self.save_game_data()

    def save_game_data(self):
        if len(self.boards) > 0:
            # Ensure status list matches the number of moves
            while len(self.status) < len(self.moves):
                self.status.append(0)  # Default to draw if missing
            
            # Clean up evaluation data to ensure no NaN values
            cleaned_eval = []
            for eval_val in self.eval:
                if eval_val is None or (isinstance(eval_val, float) and (np.isnan(eval_val) or np.isinf(eval_val))):
                    cleaned_eval.append(0)  # Default to neutral evaluation
                else:
                    try:
                        cleaned_val = float(eval_val)
                        if np.isnan(cleaned_val) or np.isinf(cleaned_val):
                            cleaned_eval.append(0)
                        else:
                            cleaned_eval.append(cleaned_val)
                    except (ValueError, TypeError):
                        cleaned_eval.append(0)
            
            # Determine overall game result for logging
            if self.status:
                final_result = self.status[0]  # White's perspective
                if self.play_against_engine:
                    if final_result == 1:
                        game_winner = "Model (White)"
                    elif final_result == -1:
                        game_winner = "Engine (Black)"
                    else:
                        game_winner = "Draw"
                else:
                    if final_result == 1:
                        game_winner = "White"
                    elif final_result == -1:
                        game_winner = "Black"
                    else:
                        game_winner = "Draw"
            else:
                game_winner = "Unknown"
            
            # add game id to data
            game_id = int(time.time() * 5000)
            game_ids = [game_id] * len(self.boards)

            # Create DataFrame with game data using cleaned evaluations
            df = pd.DataFrame({
                'boards': [b.tolist() for b in self.boards],
                'moves': [str(m) for m in self.moves],
                'eval': cleaned_eval,  # Use cleaned evaluations
                'status': self.status,
                'game_id': game_ids
            })
            
            # Append to existing CSV or create new one
            csv_filename = 'game_data.csv'
            if os.path.exists(csv_filename):
                df.to_csv(csv_filename, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_filename, index=False)
            
            print(f"Game completed with {len(self.moves)} moves - Winner: {game_winner} - Checkmates found: {self.checkmates_found} - Data saved to {csv_filename}")
            
            # Log individual game result to TensorBoard
            with self.train_summary_writer.as_default():
                # Log numeric result for TensorBoard
                result_value = 1 if game_winner == "White" else (-1 if game_winner == "Black" else 0)
                tf.summary.scalar('individual_games/game_result', result_value, step=self.game_count)
                tf.summary.scalar('individual_games/game_length', len(self.moves), step=self.game_count)
                tf.summary.scalar('individual_games/checkmates_in_game', self.checkmates_found, step=self.game_count)
                if cleaned_eval and any(e != 0 for e in cleaned_eval):
                    non_zero_evals = [e for e in cleaned_eval if e != 0]
                    if non_zero_evals:
                        tf.summary.scalar('individual_games/avg_evaluation', np.mean(non_zero_evals), step=self.game_count)
            self.train_summary_writer.flush()

    def make_move(self):
        """Make a move in the current game"""
        if self.game_over or self.board.is_game_over():
            self.game_over = True
            return
        
        # Limit game length
        if len(self.moves) >= MOVE_LIMIT:
            self.game_over = True
            # Set draw status for all moves
            self.status = [0] * len(self.moves)
            return
        
        # Save current board state
        board_state = process.board_to_array(self.board)
        self.boards.append(board_state)
        
        # Get engine evaluation before move
        evaluation = self.get_engine_evaluation()
        self.eval.append(evaluation)
        
        # Check for checkmate in current position
        if abs(evaluation) > 5000:  # Mate detected
            self.checkmates_found += 1
            print(f"Checkmate position detected! Evaluation: {evaluation}")
        
        # Determine who should move
        if self.play_against_engine:
            # Model plays white, engine plays black
            if self.board.turn:  # White's turn (model)
                move = self.get_model_move()
            else:  # Black's turn (engine)
                move = self.get_engine_move()
        else:
            # Self-play mode - model plays both sides
            move = self.get_model_move()
        
        # Make the move if valid
        if move and move in self.board.legal_moves:
            self.board.push(move)
            self.moves.append(move)
            
            # Check if game is over after the move
            if self.board.is_game_over():
                self.game_over = True
                
                # Determine game result from White's perspective
                if self.board.is_checkmate():
                    # Winner is opposite of current turn (since they just got checkmated)
                    if self.board.turn:  # White to move but checkmated
                        result = -1  # Black wins
                    else:  # Black to move but checkmated
                        result = 1   # White wins
                elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
                    result = 0  # Draw
                else:
                    result = 0  # Draw (other reasons)
                
                # Set status for all moves in the game
                self.status = [result] * len(self.moves)
                
                # Print game result
                if result == 1:
                    print(f"Game over: White wins by checkmate! ({len(self.moves)} moves)")
                elif result == -1:
                    print(f"Game over: Black wins by checkmate! ({len(self.moves)} moves)")
                else:
                    print(f"Game over: Draw! ({len(self.moves)} moves)")
        else:
            # No valid move found - this shouldn't happen
            print("No valid move found! Game ending.")
            self.game_over = True
            self.status = [0] * len(self.moves)  # Draw


if __name__ == "__main__":
    # TODO Play multiple games in same session
    # TODO Add option to play agains the model
    def top_5_accuracy(y_true, y_pred):
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)

    try:
        # Load the model
        model = tf.keras.models.load_model(
            'chess_transformer_model.keras',
            custom_objects={'top_5_accuracy': top_5_accuracy})
        
        # Initialize and start the chess game
        engine_path = "/usr/games/stockfish"  # Adjust this path to your Stockfish binary
        
        # Set play_against_engine=True to make model play against Stockfish
        # Set play_against_engine=False for self-play (default)
        game = ChessGame(model, engine_path, play_against_engine=True, parallel_games_enabled = True, parallel_games = PARALLEL_GAMES)  # Change this to switch modes
        game.resetGame()
        game.play()

    # TODO Pre train on all moves from game_data.csv
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. chess_transformer_model.keras file")
        print("2. move_to_number.pkl and number_to_move.pkl files")
        print("3. Stockfish engine installed")
        print("4. process.py module with board_to_array function")