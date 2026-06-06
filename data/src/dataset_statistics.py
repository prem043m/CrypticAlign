import pandas as pd


def generate_statistics(
    users_df,
    feedback_df
):

    stats = {

        "total_users":
            int(len(users_df)),

        "total_feedback":
            int(len(feedback_df)),

        "acceptance_rate":
            float(
                round(
                feedback_df["action"].mean(),
                3
                )
            )
    }

    pd.DataFrame(
        [stats]
    ).to_csv(
        "dataset_statistics.csv",
        index=False
    )

    print(stats)