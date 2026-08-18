from library import GameLibrary
from models import Game
import numpy as np


def extract_features(games):
    resultGames = []
    for game in games:
        extracted = [game.price,game.rating,game.playtime_forever,game.release_year]
        resultGames.append(extracted)
    return np.array(resultGames)

def scale_features(matrix: np.ndarray) -> np.ndarray:
    matrix_scaled = (matrix - np.min(matrix, axis=0))/(np.max(matrix, axis=0) - np.min(matrix,axis=0))
    return matrix_scaled

class GameRecommender:
    def __init__(self, library: GameLibrary):
        self.library = library
        self.matrix = extract_features(library.games)
        self.scaled_matrix = None

    def fit(self) -> None:
        self.scaled_matrix = scale_features(self.matrix)
        pass

    def recommend(self, game_index: int, top_k: int = 3)-> list[tuple[Game, float]]:
        diff = self.scaled_matrix - self.scaled_matrix[game_index]
        squared_diff = diff ** 2
        sum_squared = np.sum(squared_diff, axis=1)
        distances = np.sqrt(sum_squared)
        closest_indices = np.argsort(distances)[1:top_k+1]
        recommendations = []
        for idx in closest_indices:
            game = self.library.games[idx]
            dist = float(distances[idx])
            recommendations.append((game,dist))
        return recommendations

# Szybki test
library = GameLibrary.from_json("data/steam_library.json")

recommender = GameRecommender(library)
recommender.fit()

recs = recommender.recommend(game_index=1, top_k = 4)
print(f"Recommendations for game: {library.games[1].name}\n")
for game,dist in recs:
    print(f"-> {game.name} (Odległość: {dist:.2f})")






# print("Indeksy najbardziej podobnych gier:", closest_indices)



