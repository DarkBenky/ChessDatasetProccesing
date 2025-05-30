import kagglehub
import pandas as pd
import os
import chess
import chess.pgn
import numpy as np
import io
import time
import pickle
import multiprocessing as mp
from multiprocessing import Pool, Manager
from tqdm import tqdm

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

def extract_extra_moves(depth=2):
    # load moves_to_numbers.csv
    with open('move_to_number.pkl', 'rb') as f:
        move_to_number = pickle.load(f)
    with open('number_to_move.pkl', 'rb') as f:
        number_to_move = pickle.load(f)

    initial_move_count = len(move_to_number)
    print(f"Starting with {initial_move_count} known moves")

    # extract games from high_rated_games.csv
    games = pd.read_csv('high_rated_games.csv')
    total_new_moves = 0
    
    for index, row in games.iterrows():
        if index % 100 == 0:
            print(f"Processing game {index + 1}/{len(games)} - Found {total_new_moves} new moves so far")
            # periodically save progress
            with open('move_to_number.pkl', 'wb') as f:
                pickle.dump(move_to_number, f)
            with open('number_to_move.pkl', 'wb') as f:
                pickle.dump(number_to_move, f)

        board = chess.Board()
        # extract moves and add legal moves from each position
        moves = row['moves'].split()
        
        for move_idx, move in enumerate(moves):
            # Get all legal moves from current position
            legal_moves = [board.san(m) for m in board.legal_moves]  # Use SAN notation
            
            # Add any new legal moves to our dictionary
            for legal_move in legal_moves:
                if legal_move not in move_to_number:
                    move_number = len(move_to_number)
                    move_to_number[legal_move] = move_number
                    number_to_move[move_number] = legal_move
                    total_new_moves += 1
                    if total_new_moves % 1000 == 0:
                        print(f"Added {total_new_moves} new moves so far...")
            
            # Now explore legal moves to specified depth
            if depth > 0:
                explore_legal_moves_recursive(board, move_to_number, number_to_move, depth, total_new_moves)
            
            # Play the actual game move
            try:
                board.push_san(move)
            except ValueError as e:
                print(f"Invalid move {move} in game {index}: {e}")
                break  # Skip rest of this game
        

    # save the updated dictionaries
    with open('move_to_number.pkl', 'wb') as f:
        pickle.dump(move_to_number, f)
    with open('number_to_move.pkl', 'wb') as f:
        pickle.dump(number_to_move, f)
    
    final_move_count = len(move_to_number)
    print(f"Completed processing {len(games)} games")
    print(f"Total moves: {initial_move_count} → {final_move_count}")
    print(f"Added {final_move_count - initial_move_count} new unique moves")

def explore_legal_moves_recursive(board, move_to_number, number_to_move, depth, total_new_moves_ref):
    """Recursively explore legal moves to find new move patterns"""
    if depth <= 0:
        return
    
    # Make a copy of the board to avoid affecting the main game
    board_copy = board.copy()
    legal_moves = list(board_copy.legal_moves)
    
    for i, move in enumerate(legal_moves):
        # Play the move
        board_copy.push(move)
        
        # Get legal moves from this new position
        new_legal_moves = [board_copy.san(m) for m in board_copy.legal_moves]
        
        # Add any new moves we haven't seen before
        for legal_move in new_legal_moves:
            if legal_move not in move_to_number:
                move_number = len(move_to_number)
                move_to_number[legal_move] = move_number
                number_to_move[move_number] = legal_move
                # Note: We can't directly modify total_new_moves here due to scope
        
        # Recursively explore deeper (with reduced depth)
        if depth > 1:
            explore_legal_moves_recursive(board_copy, move_to_number, number_to_move, depth - 1, total_new_moves_ref)
        
        # Undo the move to try the next one
        board_copy.pop()

def extract_extra_moves_parallel(depth=2, num_processes=32, max_moves_per_level=8):
    """Parallel version using all your CPU cores"""
    # Load existing move dictionaries
    with open('move_to_number.pkl', 'rb') as f:
        move_to_number = pickle.load(f)
    with open('number_to_move.pkl', 'rb') as f:
        number_to_move = pickle.load(f)

    initial_move_count = len(move_to_number)
    print(f"Starting with {initial_move_count} known moves")

    # Load games and split into chunks for parallel processing
    games = pd.read_csv('high_rated_games.csv')
    chunk_size = max(1, len(games) // num_processes)
    game_chunks = [games[i:i + chunk_size] for i in range(0, len(games), chunk_size)]
    
    print(f"Processing {len(games)} games using {num_processes} cores")
    print(f"Split into {len(game_chunks)} chunks of ~{chunk_size} games each")

    # Use multiprocessing to process chunks in parallel with progress bar
    with Pool(processes=num_processes) as pool:
        # Create arguments for each process
        args = [(chunk, depth, i, max_moves_per_level) for i, chunk in enumerate(game_chunks)]
        
        # Process chunks in parallel with progress bar
        results = []
        with tqdm(total=len(game_chunks), desc="Processing chunks", unit="chunk") as pbar:
            for result in pool.starmap(process_game_chunk, args):
                results.append(result)
                pbar.update(1)
    
    # Merge results from all processes
    print("Merging results from all processes...")
    all_new_moves = set()
    for chunk_moves in tqdm(results, desc="Merging results", unit="chunk"):
        all_new_moves.update(chunk_moves)
    
    # Update move dictionaries with new unique moves
    print("Updating move dictionaries...")
    for move in tqdm(all_new_moves, desc="Adding new moves", unit="move"):
        if move not in move_to_number:
            move_number = len(move_to_number)
            move_to_number[move] = move_number
            number_to_move[move_number] = move
    
    # Save updated dictionaries
    print("Saving updated dictionaries...")
    with open('move_to_number.pkl', 'wb') as f:
        pickle.dump(move_to_number, f)
    with open('number_to_move.pkl', 'wb') as f:
        pickle.dump(number_to_move, f)
    
    final_move_count = len(move_to_number)
    print(f"Completed processing {len(games)} games")
    print(f"Total moves: {initial_move_count} → {final_move_count}")
    print(f"Added {final_move_count - initial_move_count} new unique moves")

def process_game_chunk(game_chunk, depth, chunk_id, max_moves_per_level=8):
    """Process a chunk of games and return new moves found"""
    new_moves = set()
    
    # Create progress bar for this chunk
    pbar = tqdm(
        total=len(game_chunk), 
        desc=f"Process {chunk_id}", 
        position=chunk_id,
        leave=False,
        unit="game"
    )
    
    for index, row in game_chunk.iterrows():
        try:
            board = chess.Board()
            moves = row['moves'].split()
            
            for move_idx, move in enumerate(moves):
                # Add legal moves from current position
                legal_moves = [board.san(m) for m in board.legal_moves]
                new_moves.update(legal_moves)
                
                # Explore legal moves to specified depth (LIMITED)
                if depth > 0:
                    explore_moves = explore_legal_moves_limited(board, depth, max_moves_per_level)
                    new_moves.update(explore_moves)
                
                # Play the actual game move
                try:
                    board.push_san(move)
                except ValueError:
                    break  # Skip rest of invalid game
                    
        except Exception as e:
            continue
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({"moves_found": len(new_moves)})
    
    pbar.close()
    return new_moves

def explore_legal_moves_limited(board, depth, max_moves_per_level=8):
    """Efficiently explore legal moves with strict limits to prevent explosion"""
    if depth <= 0:
        return set()
    
    found_moves = set()
    board_copy = board.copy()
    legal_moves = list(board_copy.legal_moves)
    
    # CRITICAL: Limit moves explored to prevent exponential explosion
    # Sort by move type to prioritize interesting moves
    legal_moves = prioritize_moves(board_copy, legal_moves)
    moves_to_explore = legal_moves[:max_moves_per_level]
    
    for move in moves_to_explore:
        board_copy.push(move)
        
        # Add legal moves from this position
        new_legal_moves = [board_copy.san(m) for m in board_copy.legal_moves]
        found_moves.update(new_legal_moves)
        
        # Recursively explore deeper (with even stricter limits)
        if depth > 1:
            deeper_moves = explore_legal_moves_limited(
                board_copy, 
                depth - 1, 
                max_moves_per_level=min(5, max_moves_per_level)  # Reduce limit at deeper levels
            )
            found_moves.update(deeper_moves)
        
        board_copy.pop()
    
    return found_moves

def prioritize_moves(board, legal_moves):
    """Prioritize interesting moves to explore first"""
    move_priorities = []
    
    for move in legal_moves:
        priority = 0
        
        # Prioritize captures
        if board.is_capture(move):
            priority += 100
        
        # Prioritize checks
        board.push(move)
        if board.is_check():
            priority += 50
        board.pop()
        
        # Prioritize promotions
        if move.promotion:
            priority += 75
        
        # Prioritize piece development (knights and bishops)
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            priority += 25
        
        # Prioritize central squares
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        if 2 <= to_file <= 5 and 2 <= to_rank <= 5:  # Central squares
            priority += 10
        
        # Add a unique identifier to break ties and make sorting stable
        move_priorities.append((priority, str(move), move))  # Added str(move) for stable sorting
    
    # Sort by priority (descending), then by move string for stability
    move_priorities.sort(key=lambda x: (-x[0], x[1]))  # Sort by -priority, then move string
    return [move for priority, move_str, move in move_priorities]

# Also add a memory-efficient streaming version for very large datasets
def extract_extra_moves_streaming(depth=2, num_processes=16, chunk_size=1000):
    """Memory-efficient version that processes games in streaming chunks"""
    with open('move_to_number.pkl', 'rb') as f:
        move_to_number = pickle.load(f)
    
    initial_count = len(move_to_number)
    print(f"Starting with {initial_count} moves")
    
    # Get total number of rows for progress tracking
    total_rows = sum(1 for _ in open('high_rated_games.csv')) - 1  # -1 for header
    total_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    # Process CSV in chunks to manage memory
    all_new_moves = set()
    
    chunk_reader = pd.read_csv('high_rated_games.csv', chunksize=chunk_size)
    
    with tqdm(total=total_chunks, desc="Processing chunks", unit="chunk") as chunk_pbar:
        for chunk_num, game_chunk in enumerate(chunk_reader, 1):
            chunk_pbar.set_description(f"Processing chunk {chunk_num}")
            
            # Split chunk for parallel processing
            mini_chunks = np.array_split(game_chunk, num_processes)
            
            with Pool(processes=num_processes) as pool:
                args = [(chunk, depth, i) for i, chunk in enumerate(mini_chunks)]
                results = pool.starmap(process_game_chunk, args)
            
            # Merge results from this chunk
            chunk_moves = set()
            for result in results:
                chunk_moves.update(result)
            all_new_moves.update(chunk_moves)
            
            chunk_pbar.update(1)
            chunk_pbar.set_postfix({
                "chunk_moves": len(chunk_moves),
                "total_moves": len(all_new_moves)
            })
    
    # Update dictionaries with all new moves
    print("Updating move dictionary...")
    for move in tqdm(all_new_moves, desc="Adding moves", unit="move"):
        if move not in move_to_number:
            move_to_number[move] = len(move_to_number)
    
    # Save updated dictionary
    with open('move_to_number.pkl', 'wb') as f:
        pickle.dump(move_to_number, f)
    
    final_count = len(move_to_number)
    print(f"Completed! Moves: {initial_count} → {final_count}")
    print(f"Added {final_count - initial_count} new moves")

# Update your main execution
if __name__ == "__main__":
    extract_extra_moves_parallel(depth=2, num_processes=16, max_moves_per_level=8)

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
