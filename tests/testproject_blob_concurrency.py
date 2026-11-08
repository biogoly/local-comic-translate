import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import patch

from app.projects import project_state_v2 as state


class ProjectBlobConcurrencyTests(unittest.TestCase):
    def test_thumbnail_and_page_readers_publish_shared_image_only_once(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            project = str(root / "page.ctpr")
            target = root / "image.png"
            pixels = b"saved image bytes"
            digest = hashlib.sha256(pixels).hexdigest()
            connection, _ = state._get_cached_connection(project)
            try:
                with connection:
                    connection.execute("INSERT INTO blobs VALUES(?, ?, ?, ?, ?)",
                                       (digest, "image", ".png", len(pixels), pixels))
                state.register_lazy_blob_path(project, str(target), digest)
                lock = state._LAZY_BLOB_LOCK
                arrived = Barrier(2)

                class ContendedLock:
                    def __enter__(self):
                        # Both consumers observed the initially absent image.
                        arrived.wait(timeout=5)
                        return lock.__enter__()

                    def __exit__(self, *args):
                        return lock.__exit__(*args)

                def read_image():
                    self.assertTrue(state.ensure_lazy_blob_materialized(str(target)))
                    return target.read_bytes()

                with (patch.object(state, "_LAZY_BLOB_LOCK", ContendedLock()),
                      patch.object(state.os, "replace", wraps=state.os.replace) as publish,
                      ThreadPoolExecutor(max_workers=2) as workers):
                    results = [workers.submit(read_image) for _ in range(2)]
                    self.assertEqual([result.result(timeout=10) for result in results], [pixels, pixels])
                    self.assertEqual(publish.call_count, 1)
            finally:
                state.close_cached_connection(project)
