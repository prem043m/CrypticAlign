# TF-IDF Vectorization
# Store vectors
# Generate feature matrix

import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from src.preprocessing.text_preprocessor import (
    clean_text
)


class TFIDFEncoder:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            max_features=5000
        )

        self.tfidf_matrix = None
        self.users_df = None

    def fit(self, users_path):

        self.users_df = pd.read_csv(
            users_path
        )

        self.users_df[
            "profile_text"
        ] = (

            self.users_df[
                "professional_summary"
            ].fillna("")
            + " "

            + self.users_df[
                "about_me"
            ].fillna("")
            + " "

            + self.users_df[
                "career_goal"
            ].fillna("")
            + " "

            + self.users_df[
                "interests"
            ].fillna("").str.replace(",", " ")
            
            + " "
            
            + self.users_df[
                "profession"
            ].fillna("")
            
            + " "
            
            + self.users_df[
                "skills"
            ].fillna("").str.replace(",", " ")
            
            + " "
            
            + self.users_df[
                "education"
            ].fillna("")
            
            + " "
            
            + self.users_df[
                "traits"
            ].fillna("").str.replace(",", " ")
            
            + " "
            
            + self.users_df[
                "networking_intent"
            ].fillna("")
        )

        self.users_df[
            "profile_text"
        ] = self.users_df[
            "profile_text"
        ].apply(clean_text)

        self.tfidf_matrix = (
            self.vectorizer.fit_transform(
                self.users_df[
                    "profile_text"
                ]
            )
        )

        return (
            self.users_df,
            self.tfidf_matrix
        )
