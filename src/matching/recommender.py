# Text Similarity
# MBTI
# Career Goal
# Location
# Experience

from sklearn.metrics.pairwise import (
    cosine_similarity
)

from src.matching.mbti_engine import (
    MBTIEngine
)


class Recommender:

    def __init__(
        self,
        users_df,
        tfidf_matrix
    ):
        self.users_df = users_df
        self.tfidf_matrix = tfidf_matrix

    def _experience_score(
        self,
        exp_a,
        exp_b
    ):

        diff = abs(exp_a - exp_b)

        if diff <= 2:
            return 100

        if diff <= 5:
            return 80

        if diff <= 10:
            return 60

        return 40

    def _location_score(
        self,
        loc_a,
        loc_b
    ):

        return 100 if loc_a == loc_b else 0

    def _career_goal_score(
        self,
        goal_a,
        goal_b
    ):

        return 100 if goal_a == goal_b else 0

    def compatibility_score(
        self,
        user_id_1,
        user_id_2
    ):

        idx1 = self.users_df[
            self.users_df["user_id"]
            == user_id_1
        ].index[0]

        idx2 = self.users_df[
            self.users_df["user_id"]
            == user_id_2
        ].index[0]

        user1 = self.users_df.iloc[idx1]
        user2 = self.users_df.iloc[idx2]

        text_similarity = (
            cosine_similarity(
                self.tfidf_matrix[idx1],
                self.tfidf_matrix[idx2]
            )[0][0]
            * 100
        )

        mbti_score = (
            MBTIEngine.get_score(
                user1["mbti"],
                user2["mbti"]
            )
        )

        career_score = (
            self._career_goal_score(
                user1["career_goal"],
                user2["career_goal"]
            )
        )

        location_score = (
            self._location_score(
                user1["location"],
                user2["location"]
            )
        )

        experience_score = (
            self._experience_score(
                user1["experience_years"],
                user2["experience_years"]
            )
        )

        final_score = (

            0.50 * text_similarity +

            0.20 * mbti_score +

            0.10 * career_score +

            0.10 * location_score +

            0.10 * experience_score
        )

        return {

            "text_similarity":
                round(text_similarity, 2),

            "mbti_score":
                round(mbti_score, 2),

            "career_goal_score":
                round(career_score, 2),

            "location_score":
                round(location_score, 2),

            "experience_score":
                round(experience_score, 2),

            "final_score":
                round(final_score, 2)
        }