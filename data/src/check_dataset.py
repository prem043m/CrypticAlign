import pandas as pd

users_df = pd.read_csv(r"data\users.csv")

print("\n=== Profession Distribution ===")
print(users_df["profession"].value_counts())

print("\n=== MBTI Distribution ===")
print(users_df["mbti"].value_counts())

print("\n=== Career Goal Distribution ===")
print(users_df["career_goal"].value_counts())

print("\n=== Sample Users ===")
print(users_df.sample(5)[[
    "user_id",
    "profession",
    "career_goal",
    "mbti",
    "interests"
]])