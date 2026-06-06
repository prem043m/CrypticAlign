import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parent.parent)
    )

from src.embeddings.tfidf_encoder import (
    TFIDFEncoder
)

from src.matching.similarity_engine import (
    SimilarityEngine
)

from src.matching.recommender import (
    Recommender
)

def main():

    users_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "users.csv"
    )

    encoder = TFIDFEncoder()
    
    Recommender = Recommender(
        users_df,
        matrix
    )
    
    users_df, matrix = (
        encoder.fit(
            users_path
        )
    )

    engine = SimilarityEngine(
        users_df,
        matrix
    )
    # verify Users
    print("\n======= U001 =======\n")
    print(
        users_df[
            users_df['user_id'] == "U001"
        ]["profile_text"].iloc[0]
    )
    print("\n===== U055 =====\n")

    print(
        users_df[
            users_df["user_id"] == "U055"
        ]["profile_text"].iloc[0]
    )
    similarity = (
        engine.get_similarity(
            "U001",
            "U055"
        )
    )

    print(
        f"Similarity: "
        f"{similarity}%"
    )

    print("\nTop Matches:\n")

    matches = (
        engine.get_top_matches(
            "U001",
            top_n=5
        )
    )

    for match in matches:

        print(
                f"""
        User ID      : {match['user_id']}
        Profession   : {match['profession']}
        MBTI         : {match['mbti']}
        Career Goal  : {match['career_goal']}
        Location     : {match['location']}
        Experience   : {match['Experience']}
        Score        : {match['score']}%
        """
            )

if __name__ == "__main__":
    main()
