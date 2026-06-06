class AdaptiveRecommender:

    def __init__(
        self,
        recommender,
        feedback_model
    ):

        self.recommender = recommender

        self.feedback_model = feedback_model
        
    def predict_match_score(
        self,
        user_id_1,
        user_id_2
    ):

        result = (
            self.recommender.compatibility_score(
                user_id_1,
                user_id_2
            )
        )

        features = [

            result["text_similarity"],

            result["mbti_score"],

            result["profession_score"],

            result["career_goal_score"],

            result["location_score"],

            result["experience_score"],

            result["skills_score"],

            result["networking_intent_score"]
        ]

        probability = (
            self.feedback_model.predict_probability(
                features
            )
        )

        return round(
            probability * 100,
            2
        )
        
    def get_candidate_pool(
        self,
        user_id,
        pool_size=30
    ):
        candidate_pool = (
            self.recommender.get_top_recommendations(
                user_id,
                top_n=pool_size
            )
        )
        return candidate_pool

    def get_top_recommendations(
        self,
        user_id,
        top_n=5
    ):
        candidate_pool = self.get_candidate_pool(
            user_id,
            pool_size=30
        )

        recommendations = []

        for candidate in candidate_pool:
            candidate_user_id = candidate["user_id"]

            ml_score = self.predict_match_score(
                user_id,
                candidate_user_id
            )

            hybrid_score = candidate["final_score"]

            final_ranking_score = 0.60 * hybrid_score + 0.40 * ml_score

            recommendations.append(
                {
                    "user_id":
                        candidate_user_id,

                    "profession":
                        candidate["profession"],

                    "career_goal":
                        candidate["career_goal"],

                    "mbti":
                        candidate["mbti"],

                    "location":
                        candidate["location"],

                    "experience":
                        candidate.get("experience"),

                    "experience_years":
                        candidate.get("experience"),

                    "hybrid_score":
                        round(hybrid_score, 2),

                    "ml_score":
                        round(ml_score, 2),

                    "score":
                        round(ml_score, 2),

                    "final_ranking_score":
                        round(final_ranking_score, 2)
                }
            )

        recommendations.sort(
            key=lambda x: x["final_ranking_score"],
            reverse=True
        )

        return recommendations[:top_n]