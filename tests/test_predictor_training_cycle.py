import tempfile
import unittest
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import predictor
from persistence import write_json_atomic


class TestPredictorTrainingCycle(unittest.TestCase):
    def test_should_retrain_without_log(self):
        original_log = predictor.TRAINING_LOG
        try:
            with tempfile.TemporaryDirectory() as tmp:
                predictor.TRAINING_LOG = f"{tmp}\\training_log.json"
                should, _reason = predictor.should_retrain(min_hours=18)
                self.assertTrue(should)
        finally:
            predictor.TRAINING_LOG = original_log

    def test_should_retrain_respects_min_hours(self):
        original_log = predictor.TRAINING_LOG
        try:
            with tempfile.TemporaryDirectory() as tmp:
                predictor.TRAINING_LOG = f"{tmp}\\training_log.json"
                now = datetime.now(timezone.utc)
                recent = now - timedelta(hours=2)
                old = now - timedelta(hours=25)

                write_json_atomic(
                    predictor.TRAINING_LOG,
                    [{"timestamp": recent.isoformat(), "metrics": {"walk_forward_f1": 51.2}}],
                )
                should_recent, _ = predictor.should_retrain(min_hours=18)
                self.assertFalse(should_recent)

                write_json_atomic(
                    predictor.TRAINING_LOG,
                    [{"timestamp": old.isoformat(), "metrics": {"walk_forward_f1": 51.2}}],
                )
                should_old, _ = predictor.should_retrain(min_hours=18)
                self.assertTrue(should_old)
        finally:
            predictor.TRAINING_LOG = original_log

    def test_resolve_training_n_jobs_defaults_to_one(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ML_TRAIN_N_JOBS", None)
            self.assertEqual(1, predictor._resolve_training_n_jobs())

    def test_resolve_training_n_jobs_invalid_falls_back_to_one(self):
        with patch.dict(os.environ, {"ML_TRAIN_N_JOBS": "abc"}, clear=False):
            self.assertEqual(1, predictor._resolve_training_n_jobs())

    def test_resolve_training_n_jobs_accepts_positive_and_minus_one(self):
        with patch.dict(os.environ, {"ML_TRAIN_N_JOBS": "2"}, clear=False):
            self.assertEqual(2, predictor._resolve_training_n_jobs())
        with patch.dict(os.environ, {"ML_TRAIN_N_JOBS": "-1"}, clear=False):
            self.assertEqual(-1, predictor._resolve_training_n_jobs())


if __name__ == "__main__":
    unittest.main()
