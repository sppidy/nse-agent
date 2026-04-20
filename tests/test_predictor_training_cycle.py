import unittest

import predictor
from tests.helpers import workspace_temp_dir


class TestPredictorTrainingCycle(unittest.TestCase):
    def test_should_retrain_without_log(self):
        original_model_pkl = predictor.MODEL_PKL
        original_model_cbm = predictor.MODEL_CBM
        try:
            with workspace_temp_dir() as tmp:
                predictor.MODEL_PKL = f"{tmp}\\predictor_catboost.pkl"
                predictor.MODEL_CBM = f"{tmp}\\predictor_catboost.cbm"
                should, _reason = predictor.should_retrain(min_hours=18)
                self.assertTrue(should)
        finally:
            predictor.MODEL_PKL = original_model_pkl
            predictor.MODEL_CBM = original_model_cbm

    def test_should_retrain_skips_when_colab_model_exists(self):
        original_model_pkl = predictor.MODEL_PKL
        original_model_cbm = predictor.MODEL_CBM
        try:
            with workspace_temp_dir() as tmp:
                predictor.MODEL_PKL = f"{tmp}\\predictor_catboost.pkl"
                predictor.MODEL_CBM = f"{tmp}\\predictor_catboost.cbm"
                with open(predictor.MODEL_PKL, "wb") as f:
                    f.write(b"model")
                should, reason = predictor.should_retrain(min_hours=18)
                self.assertFalse(should)
                self.assertIn("Colab-trained", reason)
        finally:
            predictor.MODEL_PKL = original_model_pkl
            predictor.MODEL_CBM = original_model_cbm


if __name__ == "__main__":
    unittest.main()
