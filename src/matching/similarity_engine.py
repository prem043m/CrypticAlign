# Cosine Similarity
# Top-N Similar Users
# Similarity Score

from sklearn.metrics.pairwise import (
    cosine_similarity
)


class SimilarityEngine:

    def __init__(
        self,
        users_df,
        tfidf_matrix
    ):

        self.users_df = users_df
        self.tfidf_matrix = tfidf_matrix

    def get_similarity(
        self,
        user_id_1,
        user_id_2
    ):

        idx1 = self.users_df[
            self.users_df[
                "user_id"
            ] == user_id_1
        ].index[0]

        idx2 = self.users_df[
            self.users_df[
                "user_id"
            ] == user_id_2
        ].index[0]

        similarity = cosine_similarity(
            self.tfidf_matrix[idx1],
            self.tfidf_matrix[idx2]
        )[0][0]

        return round(
            similarity * 100,
            2
        )

    def get_top_matches(
        self,
        user_id,
        top_n=5
    ):

        idx = self.users_df[
            self.users_df[
                "user_id"
            ] == user_id
        ].index[0]

        similarities = cosine_similarity(
            self.tfidf_matrix[idx],
            self.tfidf_matrix
        )[0]

        matches = []

        for i, score in enumerate(similarities):

            if i == idx:
                continue

            matches.append(
                {
                    "user_id":
                        self.users_df.iloc[i]["user_id"],

                    "profession":
                        self.users_df.iloc[i]["profession"],

                    "mbti":
                        self.users_df.iloc[i]["mbti"],

                    "career_goal":
                        self.users_df.iloc[i]["career_goal"],
    
                    "location":
                        self.users_df.iloc[i]["location"],
                    
                    "Experience":
                        self.users_df.iloc[i]["experience_years"],
                    "score":
                        round(score * 100, 2)
                }
            )

        matches.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return matches[:top_n]