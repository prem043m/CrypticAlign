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

from src.learning.feedback_dataset import (
    FeedbackDatasetBuilder
)

from src.learning.feedback_model import(
    FeedbackModel
)

from src.learning.adaptive_recommender import (
    AdaptiveRecommender
)
 
def main():

    users_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "users.csv"
    )
    feedback_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "feedback.csv"
    )

    encoder = TFIDFEncoder()
    
    users_df, matrix = (
        encoder.fit(
            users_path
        )
    )

    engine = SimilarityEngine(
        users_df,
        matrix
    )
    
    recommender = Recommender(
        users_df,
        matrix
    )
    
    feedback_builder = FeedbackDatasetBuilder(
        recommender,
        feedback_path
    )
    
    training_dataset = (
        feedback_builder.build()
    )
    
    print("\n======= Training Feedback Shape =======\n", training_dataset.shape)
    
    model = FeedbackModel()
    accuracy = (
        model.train(
            training_dataset
        )
    )
    
    print("\n======= training label check =======\n", training_dataset["label"].value_counts(normalize=True))
    adaptive = AdaptiveRecommender(
        recommender,
        model
    )
    
    for feature, coef in zip(
        training_dataset.columns[:-1],
        model.model.coef_[0]
    ):
        print(
            feature,
            round(coef,4)
        )
    
    # verify Users
    print("\n======= U005 =======\n")
    print(
        users_df[
            users_df['user_id'] == "U005"
        ]["profile_text"].iloc[0]
    )
    print("\n===== U015 =====\n")

    print(
        users_df[
            users_df["user_id"] == "U015"
        ]["profile_text"].iloc[0]
    )
    similarity = (
        engine.get_similarity(
            "U005",
            "U015"
        )
    )

    print(
        f"Similarity: "
        f"{similarity}%"
    )

    # Stage 1: Candidate Pool Generation
    candidate_pool = adaptive.get_candidate_pool("U005", pool_size=30)
    print("\n====================================================")
    print("STAGE 1: Candidate Generation")
    print("====================================================")
    print("Candidate Pool Size:", len(candidate_pool))

    # Stage 2: Top Hybrid Candidates
    print("\n====================================================")
    print("STAGE 2: TOP HYBRID CANDIDATES")
    print("====================================================")
    for cand in candidate_pool[:5]:
        print(
            f"User ID: {cand['user_id']} | "
            f"Profession: {cand['profession']} | "
            f"Hybrid Score: {cand['final_score']:.2f}%"
        )

    # Stage 3: ML Re-Ranked Recommendations
    print("\n====================================================")
    print("STAGE 3: ML RE-RANKED RECOMMENDATIONS")
    print("====================================================")
    matches = (
        adaptive.get_top_recommendations(
            "U005",
            top_n=5
        )
    )
    for match in matches:
        print(
            f"""
        User ID            : {match['user_id']}
        Profession         : {match['profession']}
        MBTI               : {match['mbti']}
        Career Goal        : {match['career_goal']}
        Location           : {match['location']}
        Experience         : {match['experience']} years
        Hybrid Score       : {match['hybrid_score']:.2f}%
        ML Score           : {match['ml_score']:.2f}%
        Final Ranking Score: {match['final_ranking_score']:.2f}%
        """
        )

    print("\n========== HYBRID SCORE (U005 -> U015) ==========\n")
    result = recommender.compatibility_score(
        "U005",
        "U015"
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )

    print("\n========== ML-BASED RANKING SCORE (U005 -> U015) ==========\n")
    predicted_prob = adaptive.predict_match_score(
        "U005",
        "U015"
    )
    print(
        f"Predicted Acceptance Probability: {predicted_prob:.2f}%"
    )

if __name__ == "__main__":
    main()
