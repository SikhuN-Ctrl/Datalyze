"""
shrink_genome_scores.py

shrink_genome_scores filters the large genome-scores.csv down to only the movies present in the movies.csv, joins in readable
tag names from genome-tags.csv and writes a much smaller and more usable file.

Reads genome-scores.csv in chunks so that the full 500MB+ file is never loaded into memory at once.

Usage: python shrink_genome_scores.py
"""

import pandas as pd
GENOME_SCORES_PATH = "genome-scores.csv"
GENOME_TAGS_PATH = "data/raw/genome-tags.csv"
MOVIES_PATH = "data/raw/movies.csv"
OUTPUT_PATH = "data/raw/genome-scores-small.csv"
CHUNK_SIZE = 500_000
MIN_RELEVANCE = 0.0

def shrink_genome_scores() -> None:
    keep_ids = set(pd.read_csv(MOVIES_PATH)["moviesId"])
    tag_names = pd.read_csv(GENOME_TAGS_PATH)

    first_chunk = True
    total_in = 0
    total_out = 0

    for chunk in pd.read_csv(GENOME_SCORES_PATH, chunksize = CHUNK_SIZE):
        total_in += len(chunk)

        filtered = chunk[chunk["movieId"].isin(keep_ids)]
        if MIN_RELEVANCE > 0:
            filtered = filtered[filtered["relevance"] >= MIN_RELEVANCE]

        filtered = filtered.merge(tag_names, on = "tagId", how = "left")

        filtered.to_csv(
            OUTPUT_PATH, 
            mode = "w" if first_chunk else "a",
            header = first_chunk,
            index = False
        )

        first_chunk = False
        total_out += len(filtered)

    print(f"Read {total_in} rows, keep {total_out} rows " f"({total_out / total_in:.1%}) -> {OUTPUT_PATH}")


if__name__ == "__main__":
    shrink_genome_scores()
