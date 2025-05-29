import kagglehub
import pandas as pd
import os
import chess
import chess.pgn
import numpy as np
import io
import time

ELO_THRESHOLD = 1650

def parse_pgn_file(pgn_file_path):
    """Parse PGN file and extract game data"""
    games_data = []
    
    print(f"Processing PGN file: {pgn_file_path}")
    start_time = time.time()
    
    # Get file size for progress tracking
    file_size = os.path.getsize(pgn_file_path)
    print(f"File size: {file_size / (1024*1024):.1f} MB")
    
    with open(pgn_file_path, 'r', encoding='utf-8') as pgn_file:
        game_count = 0
        max_games = 1_000_000  # Safety limit
        
        while game_count < max_games:
            try:
                # Get current file position for progress tracking
                current_pos = pgn_file.tell()
                progress = (current_pos / file_size) * 100 if file_size > 0 else 0
                
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    print("Reached end of file")
                    break
                
                game_count += 1
                if game_count % 1000 == 0:
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    average_time_per_game = elapsed_time / game_count
                    
                    print(f"Processed {game_count} games in {elapsed_time:.1f} seconds")
                    print(f"Progress: {progress:.1f}% through file")
                    print(f"Average time per game: {average_time_per_game:.3f} seconds")
                    print(f"Processing rate: {game_count/elapsed_time:.1f} games/second")
            
                # Extract headers
                headers = game.headers
                
                # Get moves as string
                moves = []
                board = game.board()
                for move in game.mainline_moves():
                    moves.append(board.san(move))
                    board.push(move)
                
                moves_str = ' '.join(moves)
                
                # Extract game data
                game_data = {
                    'white_username': headers.get('White', ''),
                    'black_username': headers.get('Black', ''),
                    'white_rating': headers.get('WhiteElo', ''),
                    'black_rating': headers.get('BlackElo', ''),
                    'moves': moves_str,
                    'result': headers.get('Result', ''),
                    'time_control': headers.get('TimeControl', ''),
                    'event': headers.get('Event', ''),
                    'source': 'lichess_pgn'
                }
                
                games_data.append(game_data)
    
            except Exception as e:
                print(f"Error parsing game {game_count + 1}: {e}")
                # Try to skip to next game
                continue
        
        if game_count >= max_games:
            print(f"Reached maximum game limit of {max_games}")
    
    print(f"Total games processed from PGN: {len(games_data)}")
    return pd.DataFrame(games_data)

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
    # Download Lichess dataset only
    print("Downloading Lichess dataset...")

    # Download Lichess dataset
    lichess_path = kagglehub.dataset_download("datasnaek/chess")
    print("Path to Lichess dataset files:", lichess_path)

    # Load Lichess games data
    lichess_file = os.path.join(lichess_path, 'games.csv')
    lichess_df = pd.read_csv(lichess_file)

    # Load PGN data
    pgn_file_path = 'data.pgn'
    if os.path.exists(pgn_file_path):
        pgn_df = parse_pgn_file(pgn_file_path)
    else:
        print(f"PGN file {pgn_file_path} not found. Continuing with CSV data only.")
        pgn_df = pd.DataFrame()

    # Standardize Lichess dataset
    csv_df = standardize_lichess_dataset(lichess_df)
    print(f"CSV dataset has {len(csv_df)} games")

    # Combine CSV and PGN data
    if not pgn_df.empty:
        # Ensure both dataframes have the same columns
        common_columns = ['white_username', 'black_username', 'white_rating', 'black_rating', 'moves', 'source']
        
        # Select only common columns and fill missing ones
        csv_subset = csv_df[common_columns] if all(col in csv_df.columns for col in common_columns) else csv_df
        pgn_subset = pgn_df[common_columns] if all(col in pgn_df.columns for col in common_columns) else pgn_df
        
        combined_df = pd.concat([csv_subset, pgn_subset], ignore_index=True)
        print(f"Combined dataset has {len(combined_df)} games ({len(csv_df)} from CSV + {len(pgn_df)} from PGN)")
    else:
        combined_df = csv_df
        print(f"Using CSV dataset only with {len(combined_df)} games")

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
    print(f"Dataset source: Lichess CSV + PGN")

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

    print(f"Number of Lichess games: {len(games)}")
