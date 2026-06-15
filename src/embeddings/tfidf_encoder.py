import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing.text_preprocessor import clean_text
from src.utils.model_manager import ModelManager

class TFIDFEncoder:
    """Encodes user profiles into TF‑IDF vectors.

    The encoder builds a composite ``profile_text`` column from raw user fields,
    fits a ``TfidfVectorizer`` and persists only the vectorizer. The TF‑IDF matrix
    is recomputed at runtime wherever needed, avoiding large pickle files.
    """

    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.tfidf_matrix = None
        self.users_df = None

    @staticmethod
    def build_profile_text(users_df: pd.DataFrame) -> pd.DataFrame:

        def safe_column(column_name: str):

            if column_name in users_df.columns:
                return users_df[column_name].fillna("").astype(str)

            return pd.Series(
                [""] * len(users_df),
                index=users_df.index
            )

        users_df["profile_text"] = (

            safe_column("professional_summary") + " "
            + safe_column("about_me") + " "
            + safe_column("career_goal") + " "
            + safe_column("interests").str.replace(",", " ") + " "
            + safe_column("profession") + " "
            + safe_column("skills").str.replace(",", " ") + " "
            + safe_column("education") + " "
            + safe_column("traits").str.replace(",", " ") + " "
            + safe_column("networking_intent")
        )

        users_df["profile_text"] = (
            users_df["profile_text"]
            .apply(clean_text)
        )

        return users_df

    def fit(self, users_path: str):
        """Fit the TF‑IDF vectorizer on a CSV of user profiles.

        The function loads the CSV, builds the ``profile_text`` column using the
        reusable static helper, fits the vectorizer, persists only the vectorizer,
        and returns the enriched users dataframe together with the TF‑IDF matrix.
        """
        # Load raw user data.
        self.users_df = pd.read_csv(users_path)
        # Build the concatenated text column.
        self.users_df = TFIDFEncoder.build_profile_text(self.users_df)
        # Fit vectorizer and generate TF‑IDF matrix.
        self.tfidf_matrix = self.vectorizer.fit_transform(self.users_df["profile_text"])
        # Persist only the vectorizer for reuse.
        ModelManager.save_model(self.vectorizer, "tfidf_vectorizer.pkl")
        return self.users_df, self.tfidf_matrix
