from dataclasses import dataclass

@dataclass
class Game:
    appid: int
    name: str
    genre: str
    price: float
    rating: int
    playtime_forever: int
    release_year: int

    def playtime_hours(self) -> float:
        return round(self.playtime_forever/60,1)