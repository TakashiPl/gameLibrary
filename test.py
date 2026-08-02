from library import GameLibrary

library = GameLibrary.from_json("data/steam_library.json")

print(len(library))

print(library.games[0])
print(library.games[0].playtime_hours())
print(library.filter_by_min_rating(98))
for batch in library.iter_batches(4):
    game_names = [game.name for game in batch]
    print(game_names)