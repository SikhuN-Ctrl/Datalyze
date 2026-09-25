"""
preprocessing.py

This is the data reprocessing pipeline for the Movie/TV recommendation system project.
Dataset: MovieLens ml-latest-small (ratings.csv, movies.csv, tags.csv, links.csv)

Usage: python preprocessing.py

Expects raw CSVs in data/raw/ and writes cleaned output to data/processed
"""

import os
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates/nulls and convert the UNIX timestamp to a real date."""
    ratings = ratings.drop_duplicates().dropna(subset = ["userId", "movieId", "ratings"])
    ratings ["rating_date"] = pd.to_datetime(ratings["timestamp"], unit = "s")
    return ratings


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Pull the release year of the movie/tv show title and flag out missing genres."""
    movies = movies.copy()
    movies["year"] = pd.to_numeric(
        movies["title"].str.extract(r"\((\d{4})\)\s*$",[0], errors = "coerce"))
        movies["title_clean"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex = True)
        movies["has_genres"] = movies["genres"] != "(No genres listed)"
        return movies


def build_genre_dummies(movies: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode genres: one column per genre. 0/1, indexed by movieId."""
    dummies = movies["genres"].str.get_dummies(sep = "|")
    dummies = dummies.drop(columns =["(No genres listed)"], errors = "ignore")
    dummies.insert(0, "movieId", movies["moviesId"])
    return dummies


def build_genome_features(genome_score_path: str, top_n: int = 20) ->pd.DataFrame:
    """
    Build a movie x tag feature matrix from genome-scores-small.csv. Picks from the top_n tags with the highest
    average relevance across movies and uses them as fixed columns. 
    Every movie has a value for the same set of tags which is the same shape as build_genre_dummies(), but with continuous relevance 
    scores instead of 0/1 flags.

    Optional: only called if genome_scores_path exists on disk.
    """

    genome = pd.read_csv(genome_scores_path)
    top_tags = (
        genome.groupby("tag")["relevance"].mean()
        .sort_values(ascending = False)
        .head(top_n)
        .index
    )

    genome = genome[genome["tag"].isin(top_tags)]
    return genome.pivot_table(index = "movieId", columns = "tag", values = "relevance").reset_index()


def run_pipeline(raw_dir: str = RAW_DIR, processed_dir: str = PROCESSED_DIR) -> None:
    os.makedirs(processed_dir, exist_ok = True)

    ratings = clean_ratings(pd.read_csv(os.path.join(raw_dir,"ratings.csv")))
    movies = clean_movies(pd.read_csv(os.path.join(raw_dir,"movies.csv")))
    
    merged = ratings.merge(
        movies = [["movieId", "title_clean", "year", "genres"]], on = "movieId", how = "left"
    )

    genre_dummies = build_genre_dummies(movies)

    ratings.to_csv(os.path.join(processed_dir, "ratings_clean.csv"), index = False)
    movies.to_csv(os.path.join(processed_dir, "movies_clean.csv"), index = False)
    merged.to_csv(os.path.join(processed_dir, "ratings_movies_merged.csv"), index = False)
    genre_dummies.to_csv(os.path.join(processed_dir, "genre_dummies.csv"), index = False)

    genome_path = os.path.join(raw_dir, "genome-scores-small.csv")
    if os.path.exists(genome_path):
        genome_features = build_genome_features(genome_path)
        genome_features.to_csv(os.path.join(processed_dir, "genome_features.csv"), index = False)
        print(f"Genome features: {genome_features.shape}")

    else:
        print(f"Skipped genome features: {genome_path} not found " f"(run shrink_genome_scores.py first to continue)")


    print(f"Movies: {movies.shape}, Ratings: {ratings.shape}," f"Merged: {merged.shape}, Genre: {genre_dummies.shape}")
    print(f"{(~movies['has_genres']).sum()} movies has no genres listed")
    print(f"{movies['year'].isna().sum()} titles had no parseable year")
    print(f"Saved cleaned files to: {processed_dir}")

if __name__ == "__main__":
    run_pipeline()