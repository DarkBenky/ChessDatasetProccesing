import pandas as pd
import os
import chess
import tensorflow as tf
import numpy as np
from process import construct_board_from_moves, move_to_number, number_to_move, board_to_array
from train import create_chess_transformer_model
import random
from collections import defaultdict
import pickle
import datetime

class ChessReinforcementTrainer:
    def __init__(self, model_path_1=None, model_path_2=None, learning_rate=0.0001):
        """
        Initialize the reinforcement learning trainer for chess.
        
        Args:
            model_path_1: Path to first model (if None, creates new model)
            model_path_2: Path to second model (if None, creates new model)
            learning_rate: Learning rate for training
        """
        self.num_moves = len(move_to_number)
        self.learning_rate = learning_rate
        
        # Load or create models
        if model_path_1:
            self.model_1 = tf.keras.models.load_model(model_path_1)
            print(f"Loaded model 1 from {model_path_1}")
        else:
            self.model_1 = create_chess_transformer_model(self.num_moves)
            print("Created new model 1")
            
        if model_path_2:
            self.model_2 = tf.keras.models.load_model(model_path_2)
            print(f"Loaded model 2 from {model_path_2}")
        else:
            self.model_2 = create_chess_transformer_model(self.num_moves)
            print("Created new model 2")
        
        # Compile models
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        for model in [self.model_1, self.model_2]:
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Track game history for training
        self.game_history = []
        self.results = defaultdict(int)  # Track wins/losses/draws
        
    def get_legal_move_mask(self, board):
        """Create a mask for legal moves only."""
        legal_moves = list(board.legal_moves)
        mask = np.zeros(self.num_moves, dtype=np.float32)
        
        for move in legal_moves:
            move_san = board.san(move)
            if move_san in move_to_number:
                mask[move_to_number[move_san]] = 1.0
        
        return mask
    
    def select_move(self, model, board, temperature=1.0, use_exploration=True):
        """
        Select a move using the model with optional exploration.
        
        Args:
            model: The chess model to use
            board: Current chess board state
            temperature: Temperature for move selection (higher = more random)
            use_exploration: Whether to use exploration or greedy selection
        """
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
        
        if use_exploration:
            # Apply temperature and sample
            if temperature > 0:
                scaled_predictions = np.power(masked_predictions, 1.0 / temperature)
                scaled_predictions = scaled_predictions / np.sum(scaled_predictions)
                move_idx = np.random.choice(len(scaled_predictions), p=scaled_predictions)
            else:
                move_idx = np.argmax(masked_predictions)
        else:
            # Greedy selection
            move_idx = np.argmax(masked_predictions)
        
        # Convert to chess move
        move_san = number_to_move[move_idx]
        try:
            move = board.parse_san(move_san)
            return move, move_idx, masked_predictions[move_idx]
        except ValueError:
            # Fallback: select random legal move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                random_move = random.choice(legal_moves)
                move_san = board.san(random_move)
                if move_san in move_to_number:
                    return random_move, move_to_number[move_san], 0.1
            return None, None, 0.0
    
    def play_game(self, temperature_1=1.0, temperature_2=1.0, max_moves=200):
        """
        Play a game between the two models.
        
        Returns:
            result: 1 if model_1 wins, -1 if model_2 wins, 0 for draw
            game_data: List of (board_state, move_probs, player) for training
        """
        board = chess.Board()
        game_data = []
        move_count = 0
        
        while not board.is_game_over() and move_count < max_moves:
            current_player = 1 if board.turn == chess.WHITE else 2
            current_model = self.model_1 if current_player == 1 else self.model_2
            temperature = temperature_1 if current_player == 1 else temperature_2
            
            # Store board state before move
            board_state = board_to_array(board)
            
            # Get move from current model
            move, move_idx, move_prob = self.select_move(
                current_model, board, temperature=temperature
            )
            
            if move is None or move not in board.legal_moves:
                # Invalid move - game ends, opponent wins
                result = -1 if current_player == 1 else 1
                break
            
            # Store game data for training
            game_data.append({
                'board_state': board_state.copy(),
                'move_idx': move_idx,
                'move_prob': move_prob,
                'player': current_player
            })
            
            # Make the move
            board.push(move)
            move_count += 1
        
        # Determine game result
        if board.is_game_over():
            outcome = board.outcome()
            if outcome.winner is None:
                result = 0  # Draw
            elif outcome.winner == chess.WHITE:
                result = 1  # Model 1 (White) wins
            else:
                result = -1  # Model 2 (Black) wins
        else:
            result = 0  # Draw due to move limit
        
        return result, game_data
    
    def create_training_data(self, game_data, result):
        """
        Create training data from game with rewards based on result.
        
        Args:
            game_data: List of game states and moves
            result: Game result (1, -1, or 0)
        """
        training_data_1 = []
        training_data_2 = []
        
        for data in game_data:
            board_state = data['board_state']
            move_idx = data['move_idx']
            player = data['player']
            
            # Create one-hot encoded target
            target = np.zeros(self.num_moves, dtype=np.float32)
            target[move_idx] = 1.0
            
            # Apply reward weighting
            if result == 1:  # Model 1 wins
                reward = 1.0 if player == 1 else -0.5
            elif result == -1:  # Model 2 wins
                reward = 1.0 if player == 2 else -0.5
            else:  # Draw
                reward = 0.1
            
            # Weight the target by reward (positive reinforcement for winning moves)
            if reward > 0:
                weighted_target = target * reward
            else:
                # For losing moves, reduce the probability
                weighted_target = target * abs(reward) * 0.1
            
            if player == 1:
                training_data_1.append((board_state, weighted_target))
            else:
                training_data_2.append((board_state, weighted_target))
        
        return training_data_1, training_data_2
    
    def train_on_games(self, num_games=100, batch_size=32, save_interval=50):
        """
        Train models by playing games against each other.
        
        Args:
            num_games: Number of games to play
            batch_size: Batch size for training
            save_interval: Save models every N games
        """
        print(f"Starting reinforcement training with {num_games} games...")
        
        # Track performance
        game_results = []
        training_data_1 = []
        training_data_2 = []
        
        for game_num in range(num_games):
            # Vary temperature for exploration vs exploitation
            temperature = max(0.1, 1.0 - (game_num / num_games) * 0.8)
            
            # Play a game
            result, game_data = self.play_game(
                temperature_1=temperature, 
                temperature_2=temperature
            )
            
            # Track results
            game_results.append(result)
            if result == 1:
                self.results['model_1_wins'] += 1
            elif result == -1:
                self.results['model_2_wins'] += 1
            else:
                self.results['draws'] += 1
            
            # Create training data
            data_1, data_2 = self.create_training_data(game_data, result)
            training_data_1.extend(data_1)
            training_data_2.extend(data_2)
            
            # Train models every batch_size games
            if (game_num + 1) % batch_size == 0 and training_data_1 and training_data_2:
                print(f"Training after game {game_num + 1}...")
                
                # Prepare training data for model 1
                if training_data_1:
                    X1 = np.array([data[0] for data in training_data_1])
                    y1 = np.array([data[1] for data in training_data_1])
                    self.model_1.fit(X1, y1, batch_size=min(32, len(X1)), 
                                   epochs=1, verbose=0)
                
                # Prepare training data for model 2
                if training_data_2:
                    X2 = np.array([data[0] for data in training_data_2])
                    y2 = np.array([data[1] for data in training_data_2])
                    self.model_2.fit(X2, y2, batch_size=min(32, len(X2)), 
                                   epochs=1, verbose=0)
                
                # Clear training data
                training_data_1 = []
                training_data_2 = []
                
                # Print progress
                recent_results = game_results[-batch_size:]
                wins_1 = sum(1 for r in recent_results if r == 1)
                wins_2 = sum(1 for r in recent_results if r == -1)
                draws = sum(1 for r in recent_results if r == 0)
                print(f"Last {batch_size} games - Model 1: {wins_1}, Model 2: {wins_2}, Draws: {draws}")
            
            # Save models periodically
            if (game_num + 1) % save_interval == 0:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                self.model_1.save(f'rl_model_1_game_{game_num + 1}_{timestamp}.keras')
                self.model_2.save(f'rl_model_2_game_{game_num + 1}_{timestamp}.keras')
                print(f"Saved models after game {game_num + 1}")
        
        # Final training on remaining data
        if training_data_1:
            X1 = np.array([data[0] for data in training_data_1])
            y1 = np.array([data[1] for data in training_data_1])
            self.model_1.fit(X1, y1, batch_size=min(32, len(X1)), epochs=1, verbose=0)
        
        if training_data_2:
            X2 = np.array([data[0] for data in training_data_2])
            y2 = np.array([data[1] for data in training_data_2])
            self.model_2.fit(X2, y2, batch_size=min(32, len(X2)), epochs=1, verbose=0)
        
        # Print final results
        print(f"\nFinal Results after {num_games} games:")
        print(f"Model 1 wins: {self.results['model_1_wins']}")
        print(f"Model 2 wins: {self.results['model_2_wins']}")
        print(f"Draws: {self.results['draws']}")
        
        # Save final models
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_1.save(f'rl_model_1_final_{timestamp}.keras')
        self.model_2.save(f'rl_model_2_final_{timestamp}.keras')
        
        # Save results
        with open(f'rl_results_{timestamp}.pkl', 'wb') as f:
            pickle.dump({
                'results': dict(self.results),
                'game_results': game_results
            }, f)
        
        return self.results

def main():
    """Main function to run reinforcement learning training."""
    # You can specify paths to pre-trained models here
    # If None, new models will be created
    model_1_path = 'chess_transformer_model.keras'  # Change to None for new model
    model_2_path = None  # Will create a new model
    
    # Check if model files exist
    if model_1_path and not os.path.exists(model_1_path):
        print(f"Model file {model_1_path} not found, creating new model...")
        model_1_path = None
    
    # Initialize trainer
    trainer = ChessReinforcementTrainer(
        model_path_1=model_1_path,
        model_path_2=model_2_path,
        learning_rate=0.0001
    )
    
    # Train with self-play
    results = trainer.train_on_games(
        num_games=500,  # Start with smaller number for testing
        batch_size=16,   # Train every 16 games
        save_interval=50  # Save every 50 games
    )
    
    print("Reinforcement learning training completed!")
    return trainer

if __name__ == "__main__":
    trainer = main()