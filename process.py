import chess.pgn

input_file = "lichess_db_standard_rated_2024-01.pgn"  # Replace with your actual file
output_file = "data.pgn"
min_elo = 2000

with open(input_file, encoding="utf-8") as pgn_in, open(output_file, "w", encoding="utf-8") as pgn_out:
    while True:
        game = chess.pgn.read_game(pgn_in)
        if game is None:
            break  # End of file

        try:
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
        except ValueError:
            continue  # Skip games with invalid Elo values

        if white_elo >= min_elo and black_elo >= min_elo:
            print(game, file=pgn_out, end="\n\n")
