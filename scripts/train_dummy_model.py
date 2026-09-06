"""Train a placeholder/demo recommendation-score model.

Run from the repo root:
    python scripts/train_dummy_model.py
"""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

RANDOM_SEED = 42
NUM_SAMPLES = 300

FEATURES = ["base_price", "discount_percent", "rating", "num_reviews", "stock_quantity"]
TARGET = "recommendation_score"


def generate_synthetic_data(num_samples: int, rng: np.random.Generator):
    base_price = rng.uniform(5, 500, num_samples)
    discount_percent = rng.uniform(0, 70, num_samples)
    rating = rng.uniform(0, 5, num_samples)
    num_reviews = rng.uniform(0, 2000, num_samples)
    stock_quantity = rng.uniform(0, 500, num_samples)

    noise = rng.normal(0, 0.05, num_samples)
    recommendation_score = (
        0.5
        + 0.3 * (rating / 5)
        - 0.2 * (base_price / 1000)
        + 0.1 * (discount_percent / 100)
        + noise
    )
    recommendation_score = np.clip(recommendation_score, 0, 1)

    X = np.column_stack([base_price, discount_percent, rating, num_reviews, stock_quantity])
    y = recommendation_score
    return X, y


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    X, y = generate_synthetic_data(NUM_SAMPLES, rng)

    model = LinearRegression()
    model.fit(X, y)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(repo_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "recommendation_model.pkl")
    joblib.dump(model, model_path)

    metadata = {
        "model_type": "LinearRegression",
        "features": FEATURES,
        "target": TARGET,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = os.path.join(models_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved placeholder demo model to {model_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
