import kagglehub
import pandas as pd
import os
import chess
import numpy as np

ELO_THRESHOLD = 1650

# Download Lichess dataset only
print("Downloading Lichess dataset...")

# Download Lichess dataset
lichess_path = kagglehub.dataset_download("datasnaek/chess")
print("Path to Lichess dataset files:", lichess_path)

# Load Lichess games data
lichess_file = os.path.join(lichess_path, 'games.csv')
lichess_df = pd.read_csv(lichess_file)

def standardize_lichess_dataset(lichess_df):
    """Standardize Lichess dataset column names"""
    
    # Standardize Lichess dataset columns
    lichess_standardized = lichess_df.copy()
    if 'white_username' not in lichess_standardized.columns and 'white_id' in lichess_standardized.columns:
        lichess_standardized['white_username'] = lichess_standardized['white_id']
    if 'black_username' not in lichess_standardized.columns and 'black_id' in lichess_standardized.columns:
        lichess_standardized['black_username'] = lichess_standardized['black_id']
    
    # Add source column to identify dataset origin
    lichess_standardized['source'] = 'lichess'
    
    return lichess_standardized

# Standardize Lichess dataset
combined_df = standardize_lichess_dataset(lichess_df)
print(f"Using Lichess dataset with {len(combined_df)} games")

# Filter games where both players have ratings over threshold
print("Filtering high-rated games...")
high_rated_games = combined_df[
    (pd.to_numeric(combined_df['white_rating'], errors='coerce') > ELO_THRESHOLD) & 
    (pd.to_numeric(combined_df['black_rating'], errors='coerce') > ELO_THRESHOLD)
].dropna(subset=['white_rating', 'black_rating', 'moves'])

# Remove games with empty moves
high_rated_games = high_rated_games[high_rated_games['moves'].str.len() > 0]

# Save filtered games to a new file
output_file = os.path.join('/home/user/Desktop/ChessDatasetProccesing', 'high_rated_games.csv')
high_rated_games.to_csv(output_file, index=False)

print(f"Saved {len(high_rated_games)} games with players rated over {ELO_THRESHOLD} to: {output_file}")
print(f"Dataset source: Lichess only")

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
                # Try parsing as SAN notation first (more common in chess datasets)
                chess_move = board.parse_san(move)
                board.push(chess_move)
            except ValueError:
                try:
                    # Fall back to UCI format
                    chess_move = chess.Move.from_uci(move)
                    if chess_move in board.legal_moves:
                        board.push(chess_move)
                    else:
                        print(f"Illegal UCI move: {move}")
                        break
                except ValueError:
                    print(f"Could not parse move: {move}")
                    break
        else:
            print(f"Move {move} not found in dictionary.")
            break
    
    # Convert board to a representation suitable for ML
    board_state = board_to_array(board)
    
    return board_state, next_move

def board_to_array(board):
    """Convert a chess board to a comprehensive multi-channel array representation."""
    # 12 channels for pieces (6 piece types × 2 colors)
    # + 4 channels for castling rights
    # + 1 channel for en passant
    # + 1 channel for turn to move
    # + 1 channel for move count (normalized)
    # Total: 19 channels
    
    board_array = np.zeros((8, 8, 19), dtype=np.float32)
    
    # Piece type mapping
    piece_channels = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    # Fill piece positions (12 channels)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            # White pieces: channels 0-5, Black pieces: channels 6-11
            channel = piece_channels[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            board_array[7-row, col, channel] = 1.0
    
    # Castling rights (4 channels)
    board_array[:, :, 12] = float(board.has_kingside_castling_rights(chess.WHITE))
    board_array[:, :, 13] = float(board.has_queenside_castling_rights(chess.WHITE))
    board_array[:, :, 14] = float(board.has_kingside_castling_rights(chess.BLACK))
    board_array[:, :, 15] = float(board.has_queenside_castling_rights(chess.BLACK))
    
    # En passant square (1 channel)
    if board.ep_square is not None:
        ep_row, ep_col = divmod(board.ep_square, 8)
        board_array[7-ep_row, ep_col, 16] = 1.0
    
    # Turn to move (1 channel)
    board_array[:, :, 17] = float(board.turn == chess.WHITE)
    
    # Move count normalized (1 channel)
    # Normalize by typical game length (~40 moves)
    move_count = board.fullmove_number / 40.0
    board_array[:, :, 18] = min(move_count, 1.0)
    
    return board_array

if __name__ == "__main__":
    # load the games data
    games = pd.read_csv('high_rated_games.csv')
    print(f"Number of Lichess games: {len(games)}")
