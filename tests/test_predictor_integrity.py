import tempfile
import unittest

import pandas as pd

import predictor


class TestPredictorIntegrity(unittest.TestCase):
    def test_predict_rejects_hash_mismatch(self):
        original_model_dir = predictor.MODEL_DIR
        original_hash_file = predictor.MODEL_HASH_FILE
        try:
            with tempfile.TemporaryDirectory() as tmp:
                predictor.MODEL_DIR = tmp
                predictor.MODEL_HASH_FILE = f"{tmp}\\predictor.pkl.sha256"
                model_path = f"{tmp}\\predictor.pkl"
                with open(model_path, "wb") as f:
                    f.write(b"fake-model-bytes")
                with open(predictor.MODEL_HASH_FILE, "w", encoding="utf-8") as f:
                    f.write("not-the-right-hash")

                result = predictor.predict("TEST.NS", pd.DataFrame({"Close": [1.0]}))
                self.assertIn("error", result)
                self.assertIn("integrity", result["error"].lower())
        finally:
            predictor.MODEL_DIR = original_model_dir
            predictor.MODEL_HASH_FILE = original_hash_file


if __name__ == "__main__":
    unittest.main()
