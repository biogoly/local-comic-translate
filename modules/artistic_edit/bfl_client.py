"""Optional direct BFL backend. No Diffusers imports, account service, or POST retries."""
from __future__ import annotations

import base64
from io import BytesIO
import threading
import time
from urllib.parse import urlparse
from uuid import UUID

import numpy as np
from PIL import Image
from PySide6 import QtCore
import requests

from .contracts import ArtisticEditResult


class BflCancelled(Exception):
    pass


def validate_bfl_url(url: str, *, polling=False) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or ''
    if (parsed.scheme != 'https' or parsed.username or parsed.password
            or parsed.port not in (None, 443)
            or not (host == 'bfl.ai' or host.endswith('.bfl.ai'))):
        raise ValueError('BFL returned an unexpected result host.')
    if polling and (not (host == 'api.bfl.ai' or host.startswith('api.'))
                    or parsed.path != '/v1/get_result'):
        raise ValueError('BFL returned an unexpected polling URL.')
    return url


def generate_bfl(request, api_key, cancelled, *, http=None, timeout=600, on_submitted=None):
    """Submit exactly once, poll that task, and save a validated PNG crop.

    The API key is used only on BFL API hosts, never on the result download.
    Error messages intentionally exclude request URLs/headers and response bodies.
    """
    if not api_key.strip():
        raise ValueError('Configure a Black Forest Labs API key in Provider APIs.')
    if request.lora is not None:
        raise ValueError('Local LoRAs are not supported by this BFL backend.')
    if request.seed > 2**32 - 1:
        raise ValueError('BFL seed exceeds the 32-bit range.')
    started = time.monotonic()
    deadline = started + timeout
    task_id = None
    owned_http = http is None
    http = http or requests.Session()

    def check_cancel():
        if cancelled.is_set():
            raise BflCancelled()
        if time.monotonic() >= deadline:
            raise TimeoutError('BFL task timed out; it was not resubmitted.')

    def json_request(method, url, **kwargs):
        check_cancel()
        with getattr(http, method)(url, timeout=(10, 20), allow_redirects=False, **kwargs) as response:
            if response.status_code != 200:
                raise RuntimeError(f'BFL HTTP {response.status_code}. Check your provider key, billing, and service limits.')
            return response.json()

    try:
        check_cancel()
        with Image.open(request.input_path) as image:
            image = image.convert('RGB')
            if image.size != (request.width, request.height):
                raise ValueError('BFL input dimensions changed.')
            # Tiny selections still need the service minimum of 64 pixels.
            pixels = np.asarray(image)
            pixels = np.pad(pixels, ((0, max(0, 64 - request.height)),
                                     (0, max(0, 64 - request.width)), (0, 0)), mode='edge')
            buffer = BytesIO()
            Image.fromarray(pixels).save(buffer, format='PNG')
        height, width = pixels.shape[:2]
        if width % 16 or height % 16 or width * height > 4_000_000:
            raise ValueError('BFL needs dimensions divisible by 16 and at most 4 megapixels.')
        payload = {
            'prompt': request.prompt,
            'input_image': base64.b64encode(buffer.getvalue()).decode('ascii'),
            'width': width, 'height': height,
            'seed': request.seed, 'output_format': 'png',
        }
        headers = {'x-key': api_key.strip(), 'Accept': 'application/json'}
        result = json_request('post', f'https://api.bfl.ai/v1/flux-2-{request.model.key.value}',
                              headers=headers, json=payload)
        task_id = result.get('id')
        if not isinstance(task_id, str) or not task_id or len(task_id) > 128:
            raise ValueError('BFL did not return a valid task ID. Do not resubmit automatically.')
        task_id = str(UUID(task_id))
        polling_url = validate_bfl_url(result.get('polling_url', ''), polling=True)
        if on_submitted:
            on_submitted(task_id)
        while True:
            check_cancel()
            status = json_request('get', polling_url, headers=headers)
            state = status.get('status')
            if state == 'Ready':
                sample_url = validate_bfl_url(status.get('result', {}).get('sample', ''))
                break
            if state not in ('Pending', 'Processing'):
                raise RuntimeError(f'BFL task ended without an image ({state}). Provider content restrictions may apply.')
            if cancelled.wait(1.0):
                raise BflCancelled()
        check_cancel()
        # No API credentials are attached to signed image delivery URLs.
        data = bytearray()
        with http.get(sample_url, timeout=(10, 20), allow_redirects=False, stream=True) as response:
            if response.status_code != 200:
                raise RuntimeError(f'BFL result download failed (HTTP {response.status_code}).')
            for chunk in response.iter_content(64 * 1024):
                check_cancel()
                data.extend(chunk)
                if len(data) > 32 * 1024 * 1024:
                    raise ValueError('BFL result exceeds the download limit.')
        with Image.open(BytesIO(data)) as image:
            if image.size != (width, height):
                raise ValueError('BFL returned different dimensions; the page was not changed.')
            check_cancel()
            image.convert('RGB').crop((0, 0, request.width, request.height)).save(request.output_path, format='PNG')
        check_cancel()
        return ArtisticEditResult(request.request_id, request.output_path, request.seed,
                                  time.monotonic() - started, request.width, request.height)
    except requests.RequestException:
        # POST may already have succeeded: retrying it could incur a second charge.
        suffix = f' Task ID: {task_id}.' if task_id else ''
        raise RuntimeError('BFL network error. The request was not resubmitted; check your provider dashboard before retrying.' + suffix) from None
    finally:
        if owned_http:
            http.close()


class BflEditJob(QtCore.QThread):
    result = QtCore.Signal(object)
    error = QtCore.Signal(str)
    cancelled = QtCore.Signal(str)
    submitted = QtCore.Signal(str)

    def __init__(self, request, api_key, parent=None):
        super().__init__(parent)
        self.request = request
        self._api_key = api_key
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            result = generate_bfl(self.request, self._api_key, self._cancel,
                                  on_submitted=self.submitted.emit)
            if self._cancel.is_set():
                raise BflCancelled()
            self.result.emit(result)
        except BflCancelled:
            self.cancelled.emit(self.request.request_id)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self._api_key = ''
