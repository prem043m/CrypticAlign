from src.dataset_generator import generate_users

from src.feedback_generator import generate_feedback

from src.dataset_validator import (
    validate_users,
    validate_feedback
)

from src.dataset_statistics import (
    generate_statistics
)

users_df = generate_users()

validate_users(users_df)

feedback_df = generate_feedback(
    users_df
)

validate_feedback(
    feedback_df
)

generate_statistics(
    users_df,
    feedback_df
)

print("Dataset Generation Completed")