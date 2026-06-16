import pandas as pd


class FeedbackDatasetBuilder:

    def __init__(
        self,
        recommender,
        feedback_path
    ):
        self.recommender = recommender

        self.feedback_df = pd.read_csv(
            feedback_path
        )

    def build(self):

        rows = []
        valid_user_ids = set(self.recommender.users_df["user_id"].values)

        for _, row in self.feedback_df.iterrows():
            if (
                row["user_id"] not in valid_user_ids
                or row["matched_user_id"] not in valid_user_ids
            ):
                continue

            features = (
                self.recommender.compatibility_score(
                    row["user_id"],
                    row["matched_user_id"]
                )
            )

            rows.append(
                {
                    "text_similarity":
                        features["text_similarity"],

                    "mbti_score":
                        features["mbti_score"],

                    "profession_score":
                        features["profession_score"],

                    "career_goal_score":
                        features["career_goal_score"],

                    "location_score":
                        features["location_score"],

                    "experience_score":
                        features["experience_score"],

                    "skills_score":
                        features["skills_score"],

                    "networking_intent_score":
                        features["networking_intent_score"],

                    "label":
                        row["action"]
                }
            )

        return pd.DataFrame(rows)