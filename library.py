import json
from pathlib import Path
from models import Game

class GameLibrary:
    def __init__(self,games: list[Game]) -> None:
        self.games = games

    @classmethod
    def from_json(cls, path: str | Path) -> "GameLibrary":
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        games = [Game(**r) for r in raw_data]
        return cls(games)
        
    def __len__(self) -> int:
        return len(self.games)

    def filter_by_genre(self, genre: str) -> list[Game]:
        return [game for game in self.games if game.genre.lower() == genre.lower()]

    def filter_by_min_rating(self, min_rating: int) -> list[Game]:
        return [game for game in self.games if game.rating >= min_rating]

    def iter_batches(self, batch_size: int):
        for i in range(0, len(self.games), batch_size):
            yield self.games[i : i + batch_size]