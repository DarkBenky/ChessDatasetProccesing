import kagglehub
import pandas as pd
import os
import chess
import tensorflow as tf
import numpy as np
from kagglehub import KaggleDatasetAdapter

ELO_THRESHOLD = 1650

# Download both datasets
print("Downloading datasets...")

# Download Lichess dataset
lichess_path = kagglehub.dataset_download("datasnaek/chess")
print("Path to Lichess dataset files:", lichess_path)

# Download Chess.com dataset
try:
    chesscom_df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "adityajha1504/chesscom-user-games-60000-games",
        "",
    )
    print("Chess.com dataset loaded successfully")
except Exception as e:
    print(f"Error loading Chess.com dataset: {e}")
    chesscom_df = pd.DataFrame()

# Load Lichess games data
lichess_file = os.path.join(lichess_path, 'games.csv')
lichess_df = pd.read_csv(lichess_file)

def standardize_datasets(lichess_df, chesscom_df):
    """Standardize column names and structure between datasets"""
    
    # Standardize Lichess dataset columns
    lichess_standardized = lichess_df.copy()
    if 'white_username' not in lichess_standardized.columns and 'white_id' in lichess_standardized.columns:
        lichess_standardized['white_username'] = lichess_standardized['white_id']
    if 'black_username' not in lichess_standardized.columns and 'black_id' in lichess_standardized.columns:
        lichess_standardized['black_username'] = lichess_standardized['black_id']
    
    # Add source column to identify dataset origin
    lichess_standardized['source'] = 'lichess'
    
    # Standardize Chess.com dataset columns
    if not chesscom_df.empty:
        chesscom_standardized = chesscom_df.copy()
        chesscom_standardized['source'] = 'chesscom'
        
        # Ensure required columns exist
        required_columns = ['white_username', 'black_username', 'white_rating', 'black_rating', 'moves']
        for col in required_columns:
            if col not in chesscom_standardized.columns:
                print(f"Warning: {col} not found in Chess.com dataset")
                chesscom_standardized[col] = None
    else:
        chesscom_standardized = pd.DataFrame()
    
    return lichess_standardized, chesscom_standardized

# Standardize both datasets
lichess_std, chesscom_std = standardize_datasets(lichess_df, chesscom_df)

# Combine datasets
if not chesscom_std.empty:
    # Find common columns
    common_columns = list(set(lichess_std.columns) & set(chesscom_std.columns))
    combined_df = pd.concat([
        lichess_std[common_columns], 
        chesscom_std[common_columns]
    ], ignore_index=True)
    print(f"Combined dataset created with {len(combined_df)} total games")
else:
    combined_df = lichess_std
    print(f"Using only Lichess dataset with {len(combined_df)} games")

# Filter games where both players have ratings over 1800
print("Filtering high-rated games...")
high_rated_games = combined_df[
    (pd.to_numeric(combined_df['white_rating'], errors='coerce') > ELO_THRESHOLD) & 
    (pd.to_numeric(combined_df['black_rating'], errors='coerce') > ELO_THRESHOLD)
].dropna(subset=['white_rating', 'black_rating', 'moves'])

# Save filtered games to a new file
output_file = os.path.join('/home/user/Desktop/ChessDatasetProccesing', 'high_rated_games.csv')
high_rated_games.to_csv(output_file, index=False)

print(f"Saved {len(high_rated_games)} games with players rated over 1800 to: {output_file}")
print(f"Dataset sources: {high_rated_games['source'].value_counts().to_dict()}")

# load the games data 
games = pd.read_csv('high_rated_games.csv')

# extract uniq moves
def extract_unique_moves(games):
    unique_moves = set()
    for index, row in games.iterrows():
        moves = row['moves'].split()
        for move in moves:
            unique_moves.add(move)
    return unique_moves
unique_moves = extract_unique_moves(games)
print(f"Number of unique moves: {len(unique_moves)}")

# create a dictionary to map moves to numbers
move_to_number = {move: i for i, move in enumerate(unique_moves)}
# create a dictionary to map numbers to moves
number_to_move = {i: move for i, move in enumerate(unique_moves)}

def construct_board_from_moves(moves):
    board = chess.Board()
    next_move = []
    
    for move in moves:
        if move in move_to_number:
            next_move.append(move_to_number[move])
            try:
                chess_move = chess.Move.from_uci(move)
                board.push(chess_move)
            except ValueError:
                try:
                    # Try parsing as SAN notation
                    chess_move = board.parse_san(move)
                    board.push(chess_move)
                except ValueError:
                    print(f"Could not parse move: {move}")
        else:
            print(f"Move {move} not found in dictionary.")
    
    # Convert board to a representation suitable for ML
    board_state = board_to_array(board)
    
    return board_state, next_move

def board_to_array(board):
    """Convert a chess board to a numerical array representation."""
    piece_to_int = {
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        '.': 0
    }
    
    # Create 8x8 board representation
    board_array = np.zeros((8, 8), dtype=np.int8)
    
    for i in range(8):
        for j in range(8):
            square = chess.square(j, 7-i)  # chess.square takes file, rank
            piece = board.piece_at(square)
            if piece:
                # Convert to FEN-like character and map to integer
                piece_char = piece.symbol()
                board_array[i, j] = piece_to_int[piece_char]
    
    return board_array

if __name__ == "__main__":
    # load the games data
    games = pd.read_csv('high_rated_games.csv')
    # get how many lichess games are there
    lichess_games = games[games['source'] == 'lichess']
    print(f"Number of Lichess games: {len(lichess_games)}")
    # get how many chess.com games are there
    chesscom_games = games[games['source'] == 'chesscom']
    print(f"Number of Chess.com games: {len(chesscom_games)}")
