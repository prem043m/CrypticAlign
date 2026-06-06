#!/usr/bin/env python
"""
Generate Feedback with Enhanced Learning Signals

This script regenerates feedback.csv with:
- Stronger deterministic acceptance zones
- Improved feature weighting (profession, career, interests emphasized)
- More interactions (20 candidates per user instead of 10)
- Skills and networking intent compatibility

Run this after updating the dataset to create high-quality feedback labels
for training the Logistic Regression model.
"""

import sys
from pathlib import Path

# Add data/src to path
data_src_path = Path(__file__).parent.absolute()
sys.path.insert(0, str(data_src_path))

import pandas as pd
from feedback_generator import generate_feedback

def main():
    print("=" * 80)
    print("GENERATING ENHANCED FEEDBACK DATA")
    print("=" * 80)
    
    # Load users (users.csv is in data/, which is the parent of data/src/)
    users_path = data_src_path.parent / "users.csv"
    print(f"\n[INFO] Loading users from: {users_path}")
    
    users_df = pd.read_csv(users_path)
    print(f"   - Loaded {len(users_df)} users")
    
    # Generate feedback
    print("\n[PROCESS] Generating feedback with enhanced signals...")
    print("   - Deterministic acceptance zones")
    print("   - Increased candidate pool (20 per user)")
    print("   - Improved feature weights")
    print("   - Skills & intent compatibility")
    
    feedback_df = generate_feedback(users_df)
    
    print(f"\n[SUCCESS] Generated {len(feedback_df)} feedback records")
    
    # Calculate statistics
    positive_feedback = (feedback_df["action"] == 1).sum()
    negative_feedback = (feedback_df["action"] == 0).sum()
    positive_pct = (positive_feedback / len(feedback_df)) * 100
    
    print(f"\n[STATS] Feedback Distribution:")
    print(f"   * Positive (action=1): {positive_feedback} ({positive_pct:.1f}%)")
    print(f"   * Negative (action=0): {negative_feedback} ({100-positive_pct:.1f}%)")
    
    print(f"\n[INFO] Saved to: data/feedback.csv")
    print("\n[NEXT] Next steps:")
    print("   1. python src/main.py  (to retrain model)")
    print("   2. Check feature coefficients for meaningful values")
    print("   3. Verify model accuracy improvement")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
