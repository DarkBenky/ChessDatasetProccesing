import pandas as pd
import os
import chess
import chess.svg
import tensorflow as tf
import numpy as np
from process import board_to_array
from train import load_complete_model, create_chess_transformer_model
import random
from collections import defaultdict
import pickle
import datetime
import pygame
import sys
from io import StringIO

class ChessGameVisualizer:
    def __init__(self, square_size=80):
        """Initialize pygame chess board visualizer."""
        pygame.init()
        self.square_size = square_size
        self.board_size = 8 * square_size
        self.screen = pygame.display.set_mode((self.board_size + 400, self.board_size + 100))
        pygame.display.set_caption("Chess Self-Play Visualizer")
        
        # Colors
        self.WHITE = (240, 217, 181)
        self.BLACK = (181, 136, 99)
        self.HIGHLIGHT = (255, 255, 0, 128)
        self.TEXT_COLOR = (0, 0, 0)
        self.BG_COLOR = (50, 50, 50)
        
        # Font
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Load piece images or use text
        self.piece_symbols = {
            'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
            'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
        }
        
        self.running = True
        self.clock = pygame.time.Clock()
    
    def draw_board(self, board, last_move=None, move_count=0, current_player="", game_info=""):
        """Draw the chess board with pieces."""
        self.screen.fill(self.BG_COLOR)
        
        # Draw squares
        for row in range(8):
            for col in range(8):
                color = self.WHITE if (row + col) % 2 == 0 else self.BLACK
                rect = pygame.Rect(col * self.square_size, row * self.square_size, 
                                 self.square_size, self.square_size)
                pygame.draw.rect(self.screen, color, rect)
                
                # Highlight last move
                if last_move and last_move in [chess.square(col, 7-row)]:
                    highlight_surf = pygame.Surface((self.square_size, self.square_size))
                    highlight_surf.set_alpha(128)
                    highlight_surf.fill((255, 255, 0))
                    self.screen.blit(highlight_surf, rect)
        
        # Draw pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                symbol = self.piece_symbols[piece.symbol()]
                
                # Render piece
                piece_surface = self.font.render(symbol, True, self.TEXT_COLOR)
                piece_rect = piece_surface.get_rect()
                piece_rect.center = (col * self.square_size + self.square_size // 2,
                                   (7-row) * self.square_size + self.square_size // 2)
                self.screen.blit(piece_surface, piece_rect)
        
        # Draw coordinates
        for i in range(8):
            # Files (a-h)
            file_text = self.small_font.render(chr(ord('a') + i), True, self.TEXT_COLOR)
            self.screen.blit(file_text, (i * self.square_size + 5, self.board_size - 20))
            
            # Ranks (1-8)
            rank_text = self.small_font.render(str(8-i), True, self.TEXT_COLOR)
            self.screen.blit(rank_text, (5, i * self.square_size + 5))
        
        # Draw game info
        info_x = self.board_size + 20
        y_offset = 20
        
        # Game information
        info_lines = [
            f"Move: {move_count}",
            f"Turn: {current_player}",
            game_info,
            "",
            "Controls:",
            "SPACE - Next move",
            "R - Reset game",
            "Q - Quit",
            "",
            "Press SPACE to continue..."
        ]
        
        for line in info_lines:
            if line:
                text_surface = self.small_font.render(line, True, (255, 255, 255))
                self.screen.blit(text_surface, (info_x, y_offset))
            y_offset += 25
        
        pygame.display.flip()
    
    def wait_for_input(self):
        """Wait for user input to continue."""
        waiting = True
        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    elif event.key == pygame.K_q:
                        self.running = False
                        waiting = False
                    elif event.key == pygame.K_r:
                        return 'reset'
            self.clock.tick(60)
        return 'continue'
    
    def close(self):
        """Close the visualizer."""
        pygame.quit()

class ChessReinforcementTrainer:
    def __init__(self, model_path=None, learning_rate=0.0001):
        """
        Initialize the reinforcement learning trainer for chess self-play.
        
        Args:
            model_path: Path to model directory (if None, creates new model)
            learning_rate: Learning rate for training
        """
        self.learning_rate = learning_rate
        
        # Load model and move dictionaries
        if model_path and os.path.exists(model_path):
            self.model, self.move_to_number, self.number_to_move, self.metadata = load_complete_model(model_path)
            print(f"Loaded model from {model_path}")
            
            # Create a copy of the model for self-play
            self.model_copy = tf.keras.models.clone_model(self.model)
            self.model_copy.set_weights(self.model.get_weights())
            print("Created model copy for self-play")
        else:
            # Create new models with default move dictionaries
            print("Creating new models for self-play...")
            # Load games to create move dictionaries
            games_df = pd.read_csv('high_rated_games.csv')
            unique_moves = set()
            for idx, row in games_df.iterrows():
                moves = row['moves'].split()
                for move in moves:
                    unique_moves.add(move)
            
            self.move_to_number = {move: i for i, move in enumerate(unique_moves)}
            self.number_to_move = {i: move for i, move in enumerate(unique_moves)}
            
            self.model = create_chess_transformer_model(len(self.move_to_number))
            self.model_copy = create_chess_transformer_model(len(self.move_to_number))
        
        self.num_moves = len(self.move_to_number)
        
        # Compile models
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        for model in [self.model, self.model_copy]:
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy']
            )
        
        # Load high-rated games for random positions
        self.games_df = pd.read_csv('high_rated_games.csv')
        print(f"Loaded {len(self.games_df)} high-rated games for position sampling")
        
        # Track game history and results
        self.game_history = []
        self.results = defaultdict(int)
        self.training_positions = []
        
        # Initialize visualizer
        self.visualizer = None
    
    def load_random_position(self, min_moves=10, max_moves=30):
        """Load a random position from high-rated games."""
        while True:
            try:
                # Select random game
                game_row = self.games_df.sample(1).iloc[0]
                moves_list = game_row['moves'].split()
                
                if len(moves_list) < min_moves:
                    continue
                
                # Select random position in the game
                move_count = min(len(moves_list), max_moves)
                position_idx = random.randint(min_moves, move_count - 1)
                
                # Replay moves to get board position
                board = chess.Board()
                for i in range(position_idx):
                    try:
                        move = board.parse_san(moves_list[i])
                        board.push(move)
                    except ValueError:
                        break
                
                if not board.is_game_over() and len(list(board.legal_moves)) > 3:
                    return board
                    
            except Exception:
                continue
    
    def get_legal_move_mask(self, board):
        """Create a mask for legal moves only."""
        legal_moves = list(board.legal_moves)
        mask = np.zeros(self.num_moves, dtype=np.float32)
        
        for move in legal_moves:
            move_san = board.san(move)
            if move_san in self.move_to_number:
                mask[self.move_to_number[move_san]] = 1.0
        
        return mask
    
    def select_move(self, model, board, temperature=1.0, use_exploration=True):
        """Select a move using the model with optional exploration."""
        board_state = board_to_array(board)
        board_input = board_state.reshape(1, 8, 8, 19)
        
        # Get model predictions
        predictions = model.predict(board_input, verbose=0)[0]
        
        # Mask illegal moves
        legal_mask = self.get_legal_move_mask(board)
        masked_predictions = predictions * legal_mask
        
        # Normalize predictions
        if np.sum(masked_predictions) > 0:
            masked_predictions = masked_predictions / np.sum(masked_predictions)
        else:
            # If no legal moves found in predictions, use uniform distribution over legal moves
            masked_predictions = legal_mask / np.sum(legal_mask)
        
        if use_exploration and temperature > 0:
            # Apply temperature and sample
            scaled_predictions = np.power(masked_predictions + 1e-8, 1.0 / temperature)
            scaled_predictions = scaled_predictions / np.sum(scaled_predictions)
            move_idx = np.random.choice(len(scaled_predictions), p=scaled_predictions)
        else:
            # Greedy selection
            move_idx = np.argmax(masked_predictions)
        
        # Convert to chess move
        move_san = self.number_to_move[move_idx]
        try:
            move = board.parse_san(move_san)
            if move in board.legal_moves:
                return move, move_idx, masked_predictions[move_idx]
        except ValueError:
            pass
        
        # Fallback: select random legal move
        legal_moves = list(board.legal_moves)
        if legal_moves:
            random_move = random.choice(legal_moves)
            move_san = board.san(random_move)
            if move_san in self.move_to_number:
                return random_move, self.move_to_number[move_san], 0.1
        
        return None, None, 0.0
    
    def play_self_play_game(self, temperature=1.0, max_moves=100, visualize=False, game_id=0):
        """Play a self-play game starting from a random position."""
        # Initialize visualizer if needed
        if visualize and self.visualizer is None:
            self.visualizer = ChessGameVisualizer()
        
        # Load random starting position
        board = self.load_random_position()
        original_turn = board.turn
        
        game_data = []
        move_count = 0
        game_moves = []
        
        if visualize:
            print(f"\n=== Self-Play Game {game_id} ===")
            print(f"Starting position (Turn: {'White' if board.turn else 'Black'}):")
            print(board)
            
            # Show starting position
            self.visualizer.draw_board(
                board, 
                move_count=move_count,
                current_player="White" if board.turn else "Black",
                game_info=f"Game {game_id} - Starting Position"
            )
            
            if self.visualizer.wait_for_input() == 'reset':
                return 0, [], board
            
            if not self.visualizer.running:
                return 0, [], board
        
        while not board.is_game_over() and move_count < max_moves:
            if visualize and not self.visualizer.running:
                break
                
            # Alternate models based on color and game
            if (board.turn == chess.WHITE and game_id % 2 == 0) or (board.turn == chess.BLACK and game_id % 2 == 1):
                current_model = self.model
                model_name = "Model A"
            else:
                current_model = self.model_copy
                model_name = "Model B"
            
            # Store board state before move
            board_state = board_to_array(board)
            
            # Get move from current model
            move, move_idx, move_prob = self.select_move(
                current_model, board, temperature=temperature
            )
            
            if move is None or move not in board.legal_moves:
                if visualize:
                    print(f"Invalid move from {model_name}, ending game")
                break
            
            # Store game data for training
            game_data.append({
                'board_state': board_state.copy(),
                'move_idx': move_idx,
                'move_prob': move_prob,
                'color': board.turn,
                'model': 'A' if current_model == self.model else 'B'
            })
            
            # Make the move
            move_san = board.san(move)
            game_moves.append(move_san)
            last_square = move.to_square
            board.push(move)
            move_count += 1
            
            if visualize:
                color_name = "White" if not board.turn else "Black"  # Previous player's color
                current_turn = "White" if board.turn else "Black"
                print(f"Move {move_count}: {model_name} ({color_name}) plays {move_san}")
                
                # Update visualization
                self.visualizer.draw_board(
                    board,
                    last_move=last_square,
                    move_count=move_count,
                    current_player=current_turn,
                    game_info=f"Game {game_id} - {model_name} played {move_san}"
                )
                
                # Wait for user input to continue
                user_action = self.visualizer.wait_for_input()
                if user_action == 'reset':
                    return 0, [], board
                elif not self.visualizer.running:
                    break
        
        # Determine game result
        if board.is_game_over():
            outcome = board.outcome()
            if outcome.winner is None:
                result = 0  # Draw
            elif outcome.winner == chess.WHITE:
                result = 1  # White wins
            else:
                result = -1  # Black wins
        else:
            result = 0  # Draw due to move limit
        
        if visualize:
            result_text = {1: "White wins", -1: "Black wins", 0: "Draw"}[result]
            print(f"Game result: {result_text}")
            print(f"Game moves: {' '.join(game_moves[:20])}{'...' if len(game_moves) > 20 else ''}")
            print(f"Total moves: {len(game_moves)}")
            
            # Show final position
            self.visualizer.draw_board(
                board,
                move_count=move_count,
                current_player="Game Over",
                game_info=f"Game {game_id} - {result_text}"
            )
            self.visualizer.wait_for_input()
        
        return result, game_data, board
    
    def create_training_data(self, game_data, result):
        """Create training data from self-play game with rewards."""
        training_data_a = []
        training_data_b = []
        
        for data in game_data:
            board_state = data['board_state']
            move_idx = data['move_idx']
            color = data['color']
            model = data['model']
            
            # Determine reward based on game outcome and player color
            if result == 0:  # Draw
                reward = 0.1
            elif (result == 1 and color == chess.WHITE) or (result == -1 and color == chess.BLACK):
                reward = 1.0  # Winning move
            else:
                reward = -0.5  # Losing move
            
            # Store training data
            training_example = (board_state, move_idx, reward)
            
            if model == 'A':
                training_data_a.append(training_example)
            else:
                training_data_b.append(training_example)
        
        return training_data_a, training_data_b
    
    def save_complete_model(self, filepath_prefix):
        """Save the complete model with move dictionaries and metadata."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create directory for complete model
        model_dir = f"{filepath_prefix}_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the main model
        model_path = os.path.join(model_dir, "model.keras")
        self.model.save(model_path)
        
        # Save move dictionaries
        move_dict_path = os.path.join(model_dir, "move_dictionaries.pkl")
        with open(move_dict_path, 'wb') as f:
            pickle.dump({
                'move_to_number': self.move_to_number,
                'number_to_move': self.number_to_move
            }, f)
        
        # Save metadata
        metadata_path = os.path.join(model_dir, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'num_moves': self.num_moves,
                'learning_rate': self.learning_rate,
                'results': dict(self.results),
                'save_timestamp': timestamp
            }, f)
        
        print(f"Complete model saved to {model_dir}")
        return model_dir

    def train_on_self_play(self, num_games=100, batch_size=32, save_interval=50, visualize_games=5):
        """Train models using self-play games."""
        print(f"Starting self-play training with {num_games} games...")
        print("Note: Pygame window will open for visualization. Use SPACE to advance, Q to quit, R to reset game.")
        
        # Track performance
        game_results = []
        training_data_a = []
        training_data_b = []
        
        try:
            for game_num in range(num_games):
                # Decrease temperature over time for less exploration
                temperature = max(0.1, 1.0 - (game_num / num_games) * 0.8)
                
                # Visualize first few games
                visualize = game_num < visualize_games
                
                # Play a self-play game
                result, game_data, final_board = self.play_self_play_game(
                    temperature=temperature,
                    visualize=visualize,
                    game_id=game_num
                )
                
                # Check if user quit during visualization
                if visualize and self.visualizer and not self.visualizer.running:
                    print("User quit visualization. Continuing training without visualization...")
                    visualize_games = 0  # Disable further visualization
                
                # Track results
                game_results.append(result)
                if result == 1:
                    self.results['white_wins'] += 1
                elif result == -1:
                    self.results['black_wins'] += 1
                else:
                    self.results['draws'] += 1
                
                # Create training data
                data_a, data_b = self.create_training_data(game_data, result)
                training_data_a.extend(data_a)
                training_data_b.extend(data_b)
                
                # Train models periodically
                if (game_num + 1) % batch_size == 0 and training_data_a and training_data_b:
                    print(f"Training after game {game_num + 1}...")
                    
                    # Train Model A
                    if training_data_a:
                        X_a = np.array([data[0] for data in training_data_a])
                        y_a = np.array([data[1] for data in training_data_a])
                        sample_weights_a = np.array([abs(data[2]) for data in training_data_a])
                        
                        self.model.fit(X_a, y_a, 
                                     sample_weight=sample_weights_a,
                                     batch_size=min(32, len(X_a)), 
                                     epochs=1, verbose=0)
                    
                    # Train Model B (copy)
                    if training_data_b:
                        X_b = np.array([data[0] for data in training_data_b])
                        y_b = np.array([data[1] for data in training_data_b])
                        sample_weights_b = np.array([abs(data[2]) for data in training_data_b])
                        
                        self.model_copy.fit(X_b, y_b,
                                          sample_weight=sample_weights_b,
                                          batch_size=min(32, len(X_b)), 
                                          epochs=1, verbose=0)
                    
                    # Occasionally sync models (copy A's weights to B)
                    if game_num % (batch_size * 2) == 0:
                        self.model_copy.set_weights(self.model.get_weights())
                        print("Synchronized model weights")
                    
                    # Clear training data
                    training_data_a = []
                    training_data_b = []
                    
                    # Print progress
                    recent_results = game_results[-batch_size:]
                    whites = sum(1 for r in recent_results if r == 1)
                    blacks = sum(1 for r in recent_results if r == -1)
                    draws = sum(1 for r in recent_results if r == 0)
                    print(f"Last {batch_size} games - White: {whites}, Black: {blacks}, Draws: {draws}")
                
                # Save models periodically
                if (game_num + 1) % save_interval == 0:
                    # Save complete model (recommended)
                    model_dir = self.save_complete_model(f'self_play_model_game_{game_num + 1}')
                    
                    # Also save just the keras model for backup
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    self.model.save(f'self_play_model_backup_{game_num + 1}_{timestamp}.keras')
                    print(f"Saved models after game {game_num + 1}")
        
        finally:
            # Clean up visualizer
            if self.visualizer:
                self.visualizer.close()
        
        # Print final results
        print(f"\nFinal Results after {num_games} games:")
        print(f"White wins: {self.results['white_wins']}")
        print(f"Black wins: {self.results['black_wins']}")
        print(f"Draws: {self.results['draws']}")
        
        # Save final complete model
        final_model_dir = self.save_complete_model('self_play_model_final')
        
        # Save final keras model backup
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model.save(f'self_play_model_final_backup_{timestamp}.keras')
        
        # Save detailed results
        with open(f'self_play_results_{timestamp}.pkl', 'wb') as f:
            pickle.dump({
                'results': dict(self.results),
                'game_results': game_results,
                'final_model_dir': final_model_dir
            }, f)
        
        print(f"Final complete model saved to: {final_model_dir}")
        return self.results

def main():
    """Main function to run self-play reinforcement learning."""
    # Specify path to trained model directory (from complete model save)
    model_path = None  # Set to model directory path, e.g., "chess_transformer_complete_20231201_143022"
    
    # Check for most recent complete model if no path specified
    if model_path is None:
        # Look for self-play models first, then original models
        model_dirs = [d for d in os.listdir('.') if d.startswith('self_play_model_') and os.path.isdir(d)]
        if not model_dirs:
            model_dirs = [d for d in os.listdir('.') if d.startswith('chess_transformer_complete_') and os.path.isdir(d)]
        
        if model_dirs:
            model_path = max(model_dirs)  # Get most recent
            print(f"Found model directory: {model_path}")
        else:
            print("No existing model found. Will create new model.")
    
    # Initialize trainer
    trainer = ChessReinforcementTrainer(
        model_path=model_path,
        learning_rate=0.00005  # Lower learning rate for fine-tuning
    )
    
    # Train with self-play
    results = trainer.train_on_self_play(
        num_games=200,
        batch_size=16,
        save_interval=25,  # Save more frequently
        visualize_games=10  # Visualize first 10 games with pygame
    )
    
    print("Self-play reinforcement learning completed!")
    print("Models are saved in timestamped directories for easy loading.")
    return trainer

if __name__ == "__main__":
    trainer = main()