import tempfile
import unittest
from datetime import datetime, timedelta, timezone

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


if __name__ == "__main__":
    unittest.main()
