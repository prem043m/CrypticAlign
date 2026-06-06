def validate_users(df):

    assert df["user_id"].is_unique

    assert df["age"].between(
        20,
        45
    ).all()

    assert df["experience_years"].between(
        0,
        20
    ).all()

    assert df.isnull().sum().sum() == 0

    print("Users validation passed")


def validate_feedback(df):

    assert df.isnull().sum().sum() == 0

    assert set(df["action"]) <= {0,1}

    print("Feedback validation passed")