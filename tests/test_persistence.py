import unittest
from pathlib import Path

from persistence import read_json, write_json_atomic
from tests.helpers import workspace_temp_dir


class TestPersistence(unittest.TestCase):
    def test_atomic_write_and_read(self):
        with workspace_temp_dir() as tmp:
            target = Path(tmp) / "sample.json"
            payload = {"a": 1, "b": ["x", "y"]}
            write_json_atomic(target, payload)
            loaded = read_json(target, default={})
            self.assertEqual(payload, loaded)

    def test_read_json_default_when_missing(self):
        with workspace_temp_dir() as tmp:
            missing = Path(tmp) / "missing.json"
            loaded = read_json(missing, default=[])
            self.assertEqual([], loaded)


if __name__ == "__main__":
    unittest.main()
