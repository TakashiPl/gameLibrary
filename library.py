import json
from pathlib import Path
from models import Game
import time
from functools import wraps

def measure_time(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args,**kwargs)
            end_time = time.perf_counter()
            print(end_time - start_time)
            return result
        return wrapper
class GameLibrary:
    def __init__(self,games: list[Game]) -> None:
        self.games = games

    @classmethod
    @measure_time
    def from_json(cls, path: str | Path) -> "GameLibrary":
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        games = [Game(**r) for r in raw_data]
        return cls(games)
        
    def __len__(self) -> int:
        return len(self.games)

    def iter_batches(self, batch_size: int = 50):
        for i in range(0,len(self.games), batch_size):
            yield self.games[i : i + batch_size]
