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

MOVE_LIMIT = 75
GAMES_PER_TRAINING = 10  # Train after every 10 games
LOOK_BACK = 75 # Look back N games for training

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

# Add these constants near the top of your file
SF_ANALYSIS_DEPTH_FOR_TRAINING = 8  # Depth for Stockfish analysis during training
STOCKFISH_GUIDANCE_PROB = 0.85  # Target probability for Stockfish's chosen move
PLAYED_MOVE_PROB_CAP = 0.15  # Max probability for played move if Stockfish disagrees

class ChessGame:
    def __init__(self, model, engine_path , play_against_engine=False):
        self.model = model
        self.engine = engine.SimpleEngine.popen_uci(engine_path)
        self.board = chess.Board()
        self.game_over = False
        self.play_against_engine = play_against_engine  # Store the parameter
        # data shaped as board state, move history, and game status and eval from engine
        self.boards = []
        self.moves = []
        self.eval = []
        self.status = []
        # Load move dictionaries
        with open('move_to_number.pkl', 'rb') as f:
            self.move_to_number = pickle.load(f)
        with open('number_to_move.pkl', 'rb') as f:
            self.number_to_move = pickle.load(f)
        
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
        self.game_count = 0
        self.games_per_training = GAMES_PER_TRAINING  # Train after every 5 games
        self.training_step = 0  # Track training steps for TensorBoard
        self.checkmates_found = 0  # Track checkmates found in current game
        print("ChessGame initialized with model and engine.")

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
        
        # Get all legal moves first
        legal_moves = list(self.board.legal_moves)
        legal_moves_in_dict = []
        
        # Filter legal moves that are in dictionary using algebraic notation
        for move in legal_moves:
            move_san = self.board.san(move)  # Convert to algebraic notation
            if move_san in self.move_to_number:
                move_index = self.move_to_number[move_san]
                score = predictions[0][move_index]
                
                # Calculate capture bonus with reduced aggression
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
                            
                            # Convert to a bonus between 0 and 0.2 based on value difference (reduced from 0.3)
                            if exchange_value > 0:  # Good capture (captured piece worth more)
                                # Reduce bonus from 0.05 to 0.03 per point difference
                                capture_bonus = min(0.2, exchange_value * 0.03) * randomness
                            elif exchange_value == 0:  # Equal exchange
                                capture_bonus = 0.03 * randomness  # Reduced from 0.05
                            elif exchange_value >= -2:  # Slightly bad capture but potentially tactical
                                capture_bonus = 0.01 * randomness
                            else:  # Very unfavorable - discourage most of the time
                                # Only give tiny bonus, often negative to discourage bad trades
                                capture_bonus = (0.01 - 0.02 * abs(exchange_value)) * randomness
                
                # Add the capture bonus to the prediction score
                adjusted_score = score + (capture_bonus)
                legal_moves_in_dict.append((move, move_index, adjusted_score))
    
        # If we have legal moves in dictionary
        if legal_moves_in_dict:
            # Sort by adjusted prediction probability (highest first)
            legal_moves_in_dict.sort(key=lambda x: x[2], reverse=True)
            
            # Adaptive move selection based on position evaluation
            if self.eval and abs(self.eval[-1] or 0) > 200:  # Significant advantage exists
                # In decisive positions, prefer the best move with higher probability
                top_k = min(3, len(legal_moves_in_dict))  # Less randomness in winning positions
                chosen_move = legal_moves_in_dict[np.random.randint(top_k)][0]
            else:
                # More exploration in balanced positions
                top_k = min(5, len(legal_moves_in_dict))
                chosen_move = legal_moves_in_dict[np.random.randint(top_k)][0]
            return chosen_move
        
        # Fallback: return a random legal move if no dictionary moves available
        if legal_moves:
            return legal_moves[np.random.randint(len(legal_moves))]
        
        return None

    def get_engine_move(self):
        """Get move from Stockfish engine"""
        try:
            result = self.engine.play(self.board, chess.engine.Limit(time=0.5))
            return result.move
        except:
            # Fallback to random legal move if engine fails
            legal_moves = list(self.board.legal_moves)
            return legal_moves[np.random.randint(len(legal_moves))] if legal_moves else None

    def get_engine_evaluation(self, board=None):
        """Get evaluation from Stockfish engine with proper error handling"""
        if board is None:
            board = self.board
        
        try:
            # Analyze position with timeout
            info = self.engine.analyse(board, chess.engine.Limit(time=0.15))
            score = info['score'].relative
            
            # Handle mate scores
            if score.is_mate():
                mate_in = score.mate()
                if mate_in is None:
                    return 0  # Fallback for unknown mate
                
                # Convert mate to large numeric value
                # Relative score: positive means current player is winning
                if mate_in > 0:
                    # Current player is winning in 'mate_in' moves
                    return 9999 - mate_in  # Closer mate = higher score
                else:
                    # Current player is getting mated in 'mate_in' moves
                    return -9999 - mate_in  # Closer mate = lower score
            
            # Handle regular centipawn scores
            elif score.score() is not None:
                cp_score = score.score()
                # Clamp extreme values to prevent overflow
                return max(-9999, min(9999, cp_score))
            
            else:
                # Score is None for some reason
                return 0
                
        except chess.engine.EngineTerminatedError:
            print("Engine terminated unexpectedly")
            return 0
        except chess.engine.EngineError as e:
            print(f"Engine error: {e}")
            return 0
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0

    def make_move(self):
        if self.board.is_game_over():
            return False
        
        # Determine who should move based on play mode
        if self.play_against_engine:
            if len(self.moves) % 2 == 0:  # White's turn (model)
                move = self.get_model_move()
            else:  # Black's turn (engine)
                move = self.get_engine_move()
        else:
            move = self.get_model_move()
        
        # Add null check for move
        if move is None:
            print("Warning: No valid move found, ending game")
            self.game_over = True
            return False
        
        # Get board state BEFORE making the move
        board_state = process.board_to_array(self.board)
        self.boards.append(board_state)
        
        # Convert move to algebraic notation before pushing
        try:
            move_san = self.board.san(move)
        except:
            print(f"Warning: Invalid move {move}, skipping")
            return False
    
        # Make the move on the board
        self.board.push(move)
        self.moves.append(move_san)
    
        # Get evaluation AFTER the move
        eval_score = self.get_engine_evaluation()
        self.eval.append(eval_score)
    
        # Check if game is over and determine winner
        if self.board.is_game_over():
            self.game_over = True
            
            if self.board.is_checkmate():
                # Winner is the player who just moved (opposite of current turn)
                winner = "White" if not self.board.turn else "Black"
                game_result = 1 if winner == "White" else -1
                
                # Increment checkmate counter
                self.checkmates_found += 1
            elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
                # Draw conditions
                winner = "Draw"
                game_result = 0
            else:
                # Other draw conditions
                winner = "Draw"
                game_result = 0
            
            # Assign game result to all moves from perspective of the moving player
            for i in range(len(self.moves)):
                # White moves (even indices): use game_result as is
                # Black moves (odd indices): flip the result
                if i % 2 == 0:  # White's move
                    self.status.append(game_result)
                else:  # Black's move
                    self.status.append(-game_result)
            
            print(f"Game over: {winner} wins!" if winner != "Draw" else "Game over: Draw!")
            
        # End game after 100 moves (draw by move limit)
        elif len(self.moves) >= MOVE_LIMIT:
            self.game_over = True
            winner = "Draw"
            # Assign draw status for all moves
            for i in range(len(self.moves)):
                self.status.append(0)  # Draw
            print("Game over: Draw by move limit!")
        
        return True

    def get_mate_bonus(self, board_fen, max_depth=5):
        """Check if mate is possible in next few moves and return bonus"""
        try:
            # Create temporary board from FEN
            temp_board = chess.Board(board_fen)
            current_player_is_white = temp_board.turn
            
            # Use improved evaluation method
            eval_score = self.get_engine_evaluation(temp_board)
            
            # Check if this is a mate score (very high absolute value)
            if abs(eval_score) > 9000:
                mate_in = 9999 - abs(eval_score)
                if mate_in <= max_depth:
                    # Check if mate is favorable for current player
                    mate_favors_white = eval_score > 0
                    mate_is_good = (current_player_is_white and mate_favors_white) or (not current_player_is_white and not mate_favors_white)
                    
                    if mate_is_good:
                        # Higher bonus for shorter mate sequences
                        # Mate in 1: 0.4 bonus, Mate in 2: 0.3 bonus, etc.
                        bonus = 0.5 - (mate_in - 1) * 0.1
                        return max(0.1, bonus)  # Minimum 0.1 bonus
            
            return 0.0
        except Exception as e:
            print(f"Mate bonus calculation error: {e}")
            return 0.0

    @lru_cache(maxsize=100_000)
    def _get_mate_bonus_cached(self, board_fen):
        return self.get_mate_bonus(board_fen)

    @lru_cache(maxsize=100_000)
    def _get_engine_eval_cached(self, board_fen):
        # wrap your existing engine call so it’s cached by FEN
        return self.get_engine_evaluation(chess.Board(board_fen))

    def train_model(self):
        print("Starting model training...")
        try:
            df = pd.read_csv('game_data.csv')
            latest_games = df.tail(self.games_per_training * LOOK_BACK)  # Get last N games for training

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

            for game_id, group in latest_games.groupby('game_id'):
                # per‐game counters
                game_blunders = 0
                game_mates = 0

                game_data = group.reset_index(drop=True)
                temp_cache = {}  # optional per‐game cache
                
                # Create a temporary board to track position for move conversion
                temp_board = chess.Board()

                for i, row in game_data.iterrows():
                    total_moves_processed += 1
                    board = np.array(eval(row['boards']))
                    move_str = row['moves']

                    # Convert move to SAN format if needed
                    try:
                        # First try to use the move as-is (assuming it's already SAN)
                        if move_str in move_to_number:
                            final_move_str = move_str
                        else:
                            # Try to convert from UCI to SAN
                            try:
                                uci_move = chess.Move.from_uci(move_str)
                                if uci_move in temp_board.legal_moves:
                                    final_move_str = temp_board.san(uci_move)
                                else:
                                    # Invalid move, skip
                                    moves_not_in_dict += 1
                                    continue
                            except:
                                # Not valid UCI, skip
                                moves_not_in_dict += 1
                                continue
                        
                        # Update temp_board for next iteration
                        try:
                            if move_str in move_to_number:
                                temp_board.push_san(move_str)
                            else:
                                temp_board.push(chess.Move.from_uci(move_str))
                        except:
                            # If we can't update the board, we can't continue this game
                            break
                            
                    except Exception as e:
                        print(f"Error processing move {move_str}: {e}")
                        moves_not_in_dict += 1
                        continue

                    # Handle eval scores more carefully
                    eval_score = row['eval']
                    if pd.isna(eval_score) or eval_score is None or not isinstance(eval_score, (int, float)):
                        eval_score = 0  # Default to neutral evaluation
                    else:
                        # Ensure eval_score is numeric and not NaN
                        try:
                            eval_score = float(eval_score)
                            if np.isnan(eval_score) or np.isinf(eval_score):
                                eval_score = 0
                        except (ValueError, TypeError):
                            eval_score = 0
                    
                    # Calculate move quality score
                    move_quality = 0.5  # Default neutral
                    is_blunder = False
                    
                    # Need at least two half-moves to compare same-side evals
                    if i > 1:
                        # look back two rows to get the last eval from this same mover
                        prev_eval = game_data.iloc[i-2]['eval']
                        if pd.notna(prev_eval) and isinstance(prev_eval, (int, float)):
                            try:
                                prev_eval = float(prev_eval)
                                if not np.isnan(prev_eval) and not np.isinf(prev_eval):
                                    eval_change = eval_score - prev_eval
                                    
                                    # Detect blunders (significant eval drop from player's perspective)
                                    if eval_change < -200:  # Lost more than 2 pawns worth
                                        is_blunder = True
                                        game_blunders += 1
                                        total_blunders += 1
                                        
                                        # Apply blunder penalty based on severity
                                        if eval_change < -500:  # Major blunder (5+ pawns)
                                            blunder_penalty = 0.4  # Heavy penalty
                                        elif eval_change < -300:  # Serious blunder (3-5 pawns)
                                            blunder_penalty = 0.3
                                        else:  # Minor blunder (2-3 pawns)
                                            blunder_penalty = 0.2
                                        
                                        move_quality = max(0.05, 0.5 - blunder_penalty)
                                        print(f"Blunder detected: eval change {eval_change}, penalty: -{blunder_penalty:.2f}, quality: {move_quality:.2f}")
                                    else:
                                        # Convert to probability-like score [0, 1]
                                        move_quality = 0.5 + (eval_change / 1000.0)  # Adjust scale factor
                                        move_quality = max(0.1, min(0.9, move_quality))
                            except (ValueError, TypeError):
                                pass

                    # Create probability distribution with move quality only
                    if final_move_str in move_to_number:
                        played_move_idx = move_to_number[final_move_str]
                        
                        # Create much better probability distribution
                        target_dist = np.full(n_moves, 1e-6)  # Small baseline for all moves
                        target_dist[played_move_idx] = max(0.1, move_quality)  # Ensure reasonable minimum
                        
                        # Add small probabilities to a few other legal moves (if available)
                        # This simulates exploration and makes training more stable
                        try:
                            # Get legal moves for this position
                            temp_board_copy = chess.Board(temp_board.fen())
                            temp_board_copy.pop()  # Go back one move to get position before this move
                            legal_moves_here = list(temp_board_copy.legal_moves)
                            
                            # Add small probabilities to some other legal moves
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
                        
                        # Normalize to sum to 1.0
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
                
                # Recompile model with categorical crossentropy for probability distributions
                # Use much lower learning rate
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Much lower LR
                    loss='categorical_crossentropy',
                    metrics=['accuracy', 'top_k_categorical_accuracy']
                )
                
                # Train with smaller batch size and validation
                history = self.model.fit(
                    X, y, 
                    epochs=3,  # Start with fewer epochs
                    batch_size=64,  # Much smaller batch size
                    validation_split=0.1,  # Add validation monitoring
                    verbose=1,
                    callbacks=[self.tensorboard_callback]
                )
                
                # Monitor if training is working
                val_loss = history.history.get('val_loss', [])
                if val_loss and len(val_loss) > 1:
                    if val_loss[-1] > val_loss[0]:
                        print("WARNING: Validation loss increased - consider reducing learning rate further")
                    else:
                        print(f"✅ Validation loss improved: {val_loss[0]:.4f} → {val_loss[-1]:.4f}")
                
                # Log metrics including checkmate and blunder statistics
                avg_move_quality = np.mean([np.max(target) for target in move_targets])
                self.log_custom_metrics_updated(avg_move_quality, latest_games, total_mates_found, total_blunders)
                
                # Save updated model
                self.model.save('chess_transformer_model.keras')
                print(f"Model trained on {len(boards)} positions and saved.")
                self.training_step += 1
            else:
                print("No valid training data found. Check move dictionary compatibility.")
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()

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
                
                # Fixed: Complete the checkmate games calculation
                games_with_mates = sum(1 for game_id, group in latest_games.groupby('game_id') 
                                     if any(abs(row['eval']) > 5000 for _, row in group.iterrows() 
                                           if pd.notna(row['eval']) and isinstance(row['eval'], (int, float)) and not np.isnan(row['eval'])))
                tf.summary.scalar('checkmates/games_with_mates', games_with_mates, step=self.training_step)
                
                # Log blunder statistics
                tf.summary.scalar('blunders/total_blunders_found', total_blunders, step=self.training_step)
                tf.summary.scalar('blunders/blunders_per_game', total_blunders / total_games, step=self.training_step)
                
                # Calculate games with blunders (eval drops > 200) - fixed for player perspective
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
                        # Since we're using relative scores, we can directly compare
                        # without manual perspective adjustment
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
        running = True
        auto_play = True
        move_delay = 0  # milliseconds between moves
        last_move_time = 0
        
        while running:
            current_time = pygame.time.get_ticks()
            
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
            
            # Auto-play logic
            if auto_play and not self.game_over and current_time - last_move_time > move_delay:
                self.make_move()
                last_move_time = current_time
                
                # Start new game if current one is over
                if self.game_over:
                    # time.sleep(0.1)  # Brief pause to see final position
                    self.save_game_data()
                    self.game_count += 1
                    
                    # Train model after every few games
                    if self.game_count % self.games_per_training == 0:
                        self.train_model()
                    
                    self.resetGame()
                    last_move_time = current_time
            
            # Draw everything
            self.screen.fill((255, 255, 255))
            self.draw_board()
            self.draw_info()
            
            # Draw controls
            controls = ["SPACE: Pause/Resume", "R: Reset", "ESC: Quit"]
            for i, control in enumerate(controls):
                text = self.small_font.render(control, True, self.TEXT_COLOR)
                self.screen.blit(text, (400, self.BOARD_SIZE + 10 + i * 20))
            
            pygame.display.flip()
            self.clock.tick(60)
        
        # Cleanup
        self.engine.quit()
        pygame.quit()

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


if __name__ == "__main__":
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
        game = ChessGame(model, engine_path, play_against_engine=True)  # Change this to switch modes
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