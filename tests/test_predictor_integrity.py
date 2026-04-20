import unittest

import predictor
from tests.helpers import workspace_temp_dir


class TestPredictorIntegrity(unittest.TestCase):
    def test_predict_rejects_hash_mismatch(self):
        original_model_pkl = predictor.MODEL_PKL
        original_model_cbm = predictor.MODEL_CBM
        original_hash_file = predictor.MODEL_HASH_FILE
        original_cached_model = predictor._cached_model
        try:
            with workspace_temp_dir() as tmp:
                model_path = f"{tmp}\\predictor_catboost.pkl"
                predictor.MODEL_PKL = model_path
                predictor.MODEL_CBM = f"{tmp}\\predictor_catboost.cbm"
                predictor.MODEL_HASH_FILE = f"{tmp}\\predictor_catboost.pkl.sha256"
                predictor._cached_model = None
                with open(model_path, "wb") as f:
                    f.write(b"fake-model-bytes")
                with open(predictor.MODEL_HASH_FILE, "w", encoding="utf-8") as f:
                    f.write("not-the-right-hash")

                self.assertIsNone(predictor._load_model())
        finally:
            predictor.MODEL_PKL = original_model_pkl
            predictor.MODEL_CBM = original_model_cbm
            predictor.MODEL_HASH_FILE = original_hash_file
            predictor._cached_model = original_cached_model


if __name__ == "__main__":
    unittest.main()
