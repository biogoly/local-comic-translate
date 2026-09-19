import base64
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from PIL import Image
import requests
from PySide6 import QtCore, QtWidgets
from modules.artistic_edit.bfl_client import generate_bfl, validate_bfl_url, BflCancelled
from modules.artistic_edit.contracts import ArtisticEditRequest, ArtisticEditResult, DEFAULT_MODEL_SPECS, ModelKey, DevicePolicy
from tests import testartistic_edit_controller as harness


def response(data=None, content=b'', code=200):
    result = Mock(status_code=code)
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    result.json.return_value = data
    result.iter_content.return_value = [content]
    return result


class BflAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        Image.new('RGB', (64, 64), 'white').save(root / 'input.png')
        self.request = ArtisticEditRequest(str(uuid4()), str(root / 'input.png'), str(root / 'output.png'),
            'Replace the title', 64, 64, 4, 1.0, 42, DEFAULT_MODEL_SPECS[ModelKey.KLEIN_4B], None, DevicePolicy.CPU)
        self.http = Mock()
        self.task_id = str(uuid4())
        self.http.post.return_value = response({'id': self.task_id,
            'polling_url': 'https://api.us1.bfl.ai/v1/get_result?id=' + self.task_id})
        self.poll = {'status': 'Ready', 'result': {'sample': 'https://delivery-us1.bfl.ai/result.png?signature=secret'}}
        self.buffer = BytesIO()
        Image.new('RGB', (64, 64), 'red').save(self.buffer, format='PNG')
        self.http.get.side_effect = [response(self.poll), response(content=self.buffer.getvalue())]
        self.cancel = threading.Event()

    def generate(self, **kwargs):
        return generate_bfl(self.request, 'provider-secret', self.cancel, http=self.http, **kwargs)

    def test_one_submit_payload_dimensions_key_is_not_sent_to_download(self):
        result = self.generate()
        self.http.post.assert_called_once()
        call = self.http.post.call_args
        self.assertEqual(call.args[0], 'https://api.bfl.ai/v1/flux-2-klein-4b')
        self.assertNotIn('steps', call.kwargs['json'])
        self.assertNotIn('lora', call.kwargs['json'])
        encoded = call.kwargs['json']['input_image']
        self.assertEqual(Image.open(BytesIO(base64.b64decode(encoded))).size, (64, 64))
        self.assertNotIn('headers', self.http.get.call_args_list[-1].kwargs)
        self.assertFalse(self.http.get.call_args_list[-1].kwargs['allow_redirects'])
        self.assertEqual(Image.open(result.output_path).getpixel((0, 0)), (255, 0, 0))

    def test_9b_endpoint(self):
        self.request = replace(self.request, model=DEFAULT_MODEL_SPECS[ModelKey.KLEIN_9B])
        self.generate()
        self.assertTrue(self.http.post.call_args.args[0].endswith('flux-2-klein-9b'))

    def test_uncertain_submission_does_not_retry_or_expose_secret(self):
        self.http.post.side_effect = requests.Timeout('provider-secret')
        with self.assertRaisesRegex(RuntimeError, 'not resubmitted') as error:
            self.generate()
        self.assertNotIn('provider-secret', str(error.exception))
        self.http.post.assert_called_once()
        self.http.get.assert_not_called()

    def test_cancel_before_submit_makes_no_network_request(self):
        self.cancel.set()
        with self.assertRaises(BflCancelled):
            self.generate()
        self.http.post.assert_not_called()

    def test_cancel_after_submission_does_not_poll_or_write_result(self):
        with self.assertRaises(BflCancelled):
            self.generate(on_submitted=lambda _id: self.cancel.set())
        self.http.post.assert_called_once()
        self.http.get.assert_not_called()
        self.assertFalse(Path(self.request.output_path).exists())

    def test_polling_timeout_does_not_resubmit(self):
        with patch('modules.artistic_edit.bfl_client.time.monotonic', side_effect=[0, 0, 0, 601]):
            with self.assertRaisesRegex(TimeoutError, 'not resubmitted'):
                self.generate()
        self.http.post.assert_called_once()

    def test_poll_network_failure_reports_task_id_without_retry(self):
        self.http.get.side_effect = requests.ConnectionError('secret transport details')
        with self.assertRaises(RuntimeError) as error:
            self.generate()
        self.assertIn(self.task_id, str(error.exception))
        self.assertNotIn('secret transport details', str(error.exception))
        self.http.post.assert_called_once()

    def test_http_key_and_billing_errors_never_retry(self):
        for code in (401, 402, 429, 500):
            with self.subTest(code=code):
                self.http.post.reset_mock()
                self.http.post.return_value = response(code=code)
                with self.assertRaisesRegex(RuntimeError, str(code)):
                    self.generate()
                self.http.post.assert_called_once()

    def test_moderation_error_does_not_retry(self):
        self.http.get.side_effect = [response({'status': 'Content Moderated'})]
        with self.assertRaisesRegex(RuntimeError, 'restrictions'):
            self.generate()
        self.http.post.assert_called_once()
        self.assertFalse(Path(self.request.output_path).exists())

    def test_malformed_result_dimensions_do_not_get_written(self):
        buffer = BytesIO()
        Image.new('RGB', (128, 64)).save(buffer, format='PNG')
        self.http.get.side_effect = [response(self.poll), response(content=buffer.getvalue())]
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            self.generate()
        self.assertFalse(Path(self.request.output_path).exists())

    def test_unexpected_hosts_and_redirects_cannot_receive_key(self):
        for url in ('http://api.bfl.ai/v1/get_result', 'https://api.bfl.ai.evil.test/v1/get_result',
                    'https://user@api.bfl.ai/v1/get_result', 'https://api.bfl.ai:8443/v1/get_result',
                    'https://delivery-us1.bfl.ai/v1/get_result'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_bfl_url(url, polling=True)
        self.http.post.return_value = response(code=302)
        with self.assertRaisesRegex(RuntimeError, '302'):
            self.generate()
        self.http.get.assert_not_called()

    def test_api_keys_and_invalid_seed_validated_before_upload(self):
        self.request = replace(self.request, seed=2**40)
        with self.assertRaisesRegex(ValueError, '32-bit'):
            self.generate()
        self.http.post.assert_not_called()

    def test_small_crop_is_padded_then_restored(self):
        Image.new('RGB', (16, 32)).save(self.request.input_path)
        self.request = replace(self.request, width=16, height=32)
        result = self.generate()
        with Image.open(result.output_path) as image:
            self.assertEqual(image.size, (16, 32))


class BflControllerTests(unittest.TestCase):
    setUpClass = classmethod(harness.ArtisticEditControllerTests.setUpClass.__func__)
    setUp = harness.ArtisticEditControllerTests.setUp
    _mark_window_update = harness.ArtisticEditControllerTests._mark_window_update
    _session = harness.ArtisticEditControllerTests._session

    def cloud_setup(self):
        self.main.settings_page = SimpleNamespace(get_credentials=lambda service: {'api_key': 'test-key'})
        self.panel.backend_combo.setCurrentIndex(1)
        self.panel.area_combo.setCurrentIndex(2)
        self.panel.set_prompt('Edit title')
        self.controller.configured_worker_python = Mock(side_effect=AssertionError('Local runtime must not be needed'))
        self.controller.update_availability()

    def test_cloud_needs_no_local_runtime_and_disables_local_controls(self):
        self.cloud_setup()
        self.assertTrue(self.panel.generate_button.isEnabled())
        for widget in (self.panel.lora_combo, self.panel.steps_spin, self.panel.device_combo,
                       self.panel.guidance_spin, self.panel.offload_gpu_spin):
            self.assertFalse(widget.isEnabled())
        self.assertEqual(self.panel.options().lora_index, -1)
        self.panel.backend_combo.setCurrentIndex(0)
        self.assertTrue(self.panel.device_combo.isEnabled())
        self.assertTrue(self.panel.steps_spin.isEnabled())

    def test_inactive_local_worker_error_cannot_discard_cloud_job(self):
        session = self._session()
        session.backend = 'bfl'
        self.controller._session = session
        self.controller._on_worker_crashed('old local worker exited')
        self.controller._on_worker_error('old', 'failed', '')
        self.controller._on_cancelled(str(uuid4()))
        self.assertIs(self.controller._session, session)

    def test_declining_upload_makes_no_job(self):
        self.cloud_setup()
        with patch('app.controllers.artistic_edit.QtWidgets.QMessageBox.question', return_value=QtWidgets.QMessageBox.No), \
             patch('app.controllers.artistic_edit.BflEditJob') as job:
            self.controller.generate_preview(confirm_whole_page=False)
        job.assert_not_called()
        self.assertIsNone(self.controller._session)

    def test_approved_cloud_dispatch_creates_job_and_busy_state(self):
        self.cloud_setup()
        with patch('app.controllers.artistic_edit.QtWidgets.QMessageBox.question', return_value=QtWidgets.QMessageBox.Yes), \
             patch('app.controllers.artistic_edit.BflEditJob') as job:
            self.controller.generate_preview(confirm_whole_page=False)
        job.assert_called_once()
        request, key, _parent = job.call_args.args
        self.assertEqual(key, 'test-key')
        self.assertTrue(Path(request.input_path).is_file())
        self.assertIsNone(request.lora)
        self.assertEqual(self.controller._session.backend, 'bfl')
        self.assertTrue(self.panel.is_busy())
        self.controller._bfl_job = None

    def test_cloud_apply_keeps_undo_and_metadata_without_key(self):
        session = self._session()
        session.backend = 'bfl'
        self.controller._session = session
        self.controller._apply_session(session)
        self.assertEqual(self.stack.count(), 1)
        record = self.main.image_patches[self.page_path][0]
        self.assertEqual(record['metadata']['backend'], 'bfl')
        self.assertNotIn('api_key', record['metadata'])
        self.assertIsNone(record['metadata']['steps'])
        self.stack.undo()
        self.assertEqual(self.main.image_patches.get(self.page_path, []), [])
        self.stack.redo()
        self.assertEqual(len(self.main.image_patches[self.page_path]), 1)

    def test_background_job_delivers_preview_on_gui_thread(self):
        self.cloud_setup()
        seen_threads = []

        def fake_generation(request, key, cancelled, **kwargs):
            Image.new('RGB', (request.width, request.height), 'blue').save(request.output_path)
            return ArtisticEditResult(request.request_id, request.output_path, request.seed, 0.1,
                                      request.width, request.height)

        def preview(*args, **kwargs):
            seen_threads.append(QtCore.QThread.currentThread())
            return harness._PreviewDecision(QtWidgets.QDialog.Rejected)

        self.controller.preview_factory = preview
        with patch('app.controllers.artistic_edit.QtWidgets.QMessageBox.question', return_value=QtWidgets.QMessageBox.Yes), \
             patch('modules.artistic_edit.bfl_client.generate_bfl', side_effect=fake_generation):
            self.controller.generate_preview(confirm_whole_page=False)
            timer = QtCore.QElapsedTimer()
            timer.start()
            while self.panel.is_busy() and timer.elapsed() < 3000:
                self.app.processEvents()
        self.assertFalse(self.panel.is_busy())
        self.assertEqual(seen_threads, [self.app.thread()])
        self.assertIsNone(self.controller._bfl_job)
        self.assertIsNone(self.controller._session)
        self.assertEqual(self.stack.count(), 0)


if __name__ == '__main__':
    unittest.main()
