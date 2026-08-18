from library import GameLibrary
from recommender import d

library = GameLibrary.from_json("data/steam_library.json")

print(len(library))

print(library.games[0])
print(library.games[0].playtime_hours())
for batch in library.iter_batches(3):
    game_names = [game.name for game in batch]
    print(game_names)