from stockfish import Stockfish

# Print the docstrings for the methods we're interested in
print("set_elo_rating docstring:")
print(Stockfish.set_elo_rating.__doc__)
print("\nset_skill_level docstring:")
print(Stockfish.set_skill_level.__doc__)

# Create a basic instance
stockfish = Stockfish()

# Print available parameters
print("\nAvailable parameters:")
print(stockfish.get_parameters()) 