"""Release runtime lookup must work without historical experiment directories."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.bunny_runtime import runtime_provenance


class ReleaseRuntimeTests(unittest.TestCase):
    def test_adjacent_provenance_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root/'src'
            dll = package/'uipc/_native/uipc_backend_cuda.dll'
            dll.parent.mkdir(parents=True)
            dll.write_bytes(b'fixture DLL bytes, never loaded')
            record = {'release_source_build': True,
                      'dll_sha256': hashlib.sha256(dll.read_bytes()).hexdigest(),
                      'event_fallback_control': True}
            (root/'provenance.json').write_text(json.dumps(record), encoding='utf-8')
            self.assertEqual(runtime_provenance(package), record)
            dll.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'differs from'):
                runtime_provenance(package)

    def test_missing_runtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Missing CUDA library'):
                runtime_provenance(Path(directory)/'src')


if __name__ == '__main__':
    unittest.main()
