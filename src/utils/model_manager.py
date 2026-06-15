from pathlib import Path
import json
import joblib


MODEL_DIR = (
    Path(__file__).resolve()
    .parent.parent.parent
    / "models"
)

MODEL_DIR.mkdir(
    exist_ok=True
)


class ModelManager:

    @staticmethod
    def save_model(
        model,
        filename
    ):

        path = MODEL_DIR / filename

        joblib.dump(
            model,
            path
        )

    @staticmethod
    def load_model(
        filename
    ):

        path = MODEL_DIR / filename

        if not path.exists():
            return None

        return joblib.load(path)

    @staticmethod
    def save_metadata(
        metadata,
        filename="model_metadata.json"
    ):

        path = MODEL_DIR / filename

        with path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    @staticmethod
    def load_metadata(
        filename="model_metadata.json"
    ):

        path = MODEL_DIR / filename

        if not path.exists():
            return {}

        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
