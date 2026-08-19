import pandas as pd

df = pd.read_json("data/steam_library.json")

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATAFRAME INFO ---")
print(df.info())

class SteamDataPipeline:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_json(self.file_path)
        return self.df

    def transform(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Firstly load your data using load_data()!")

        self.df["playtime_hours"] = (self.df["playtime_forever"]/60).round(1)
        self.df["is_free"] = self.df["price"] == 0

        return self.df

    def get_genre_stats(self) -> pd.DataFrame:
        return self.df.groupby("genre").agg(
            game_count=("appid","count"),
            avg_price=("price","mean"),
            avg_playtime_hours=("playtime_hours","mean"),
            avg_rating=("rating","mean")
        ).reset_index().sort_values(by="game_count", ascending=False)

    def get_stats(self) -> pd.DataFrame:
        return self.df.describe()

    def export_clean_data(self, output_path: str = "data/steam_clean.csv") -> None:
        if self.df is None:
            raise ValueError("No data to save! Firstly use the load_data() and transform()")

        self.df.to_csv(output_path,index=False)
        print(f"Data successfully saved at: {output_path}")

if __name__ == "__main__":
    pipeline = SteamDataPipeline("data/steam_library.json")
    pipeline.load_data()
    df_clean = pipeline.transform()

    print("--- PROCESSED DATA (FIRST 5 ROWS) ---")
    print(df_clean[["name","price","is_free","playtime_hours"]].head())

    print("\n--- DESCRIBE ---")
    print(pipeline.get_stats())

    print("\n--- STATYSTYKI GATUNKÓW (GROUPBY) ---")
    print(pipeline.get_genre_stats())

    pipeline.export_clean_data("data/steam_clean.csv")