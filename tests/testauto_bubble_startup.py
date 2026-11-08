"""Pending canvas edits reach page protection before an automatic worker starts."""
import os
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from app.controllers.text import TextController
from app.ui.canvas.webtoons.coordinate_converter import CoordinateConverter


class AutomaticBubbleStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        # Import the real controller without its import-time model setup.
        import modules.utils.download as downloads
        setup_models = downloads.ensure_mandatory_models
        with patch.object(downloads, "ensure_mandatory_models"):
            import controller
        controller.ensure_mandatory_models = setup_models
        cls.controller_module = controller

    def run_startup(self, *, webtoon=False, enabled=True):
        events = []
        committed = {"value": None}
        pages = ["page.png", "other.png"]
        states = {
            path: {"target_lang": "English", "viewer_state": {"text_items_state": []}}
            for path in pages
        }

        def flush():
            events.append("flush")
            committed["value"] = "last pending edit"

        def save_current():
            events.append("save_current")
            states["page.png"]["viewer_state"]["text_items_state"] = [{"text": committed["value"]}]

        def save_all():
            events.append("save_all")
            for path in pages:
                states[path]["viewer_state"]["text_items_state"] = [{"text": committed["value"] + " " + path}]

        worker_snapshots = []

        def start_worker(function, *_args):
            events.append("worker")
            worker_snapshots.append(deepcopy(states))
            function()

        main = SimpleNamespace(
            image_files=pages, image_states=states, webtoon_mode=webtoon,
            render_settings=lambda: SimpleNamespace(auto_bubble_fit=enabled),
            text_ctrl=SimpleNamespace(_commit_pending_text_command=Mock(side_effect=flush)),
            image_ctrl=SimpleNamespace(save_current_image_state=Mock(side_effect=save_current),
                                       clear_page_skip_errors_for_paths=Mock()),
            image_viewer=SimpleNamespace(webtoon_manager=SimpleNamespace(scene_item_manager=SimpleNamespace(
                save_all_scene_items_to_states=Mock(side_effect=save_all)))),
            _start_batch_report=Mock(), manual_radio=SimpleNamespace(isChecked=lambda: False),
            translate_button=Mock(), cancel_button=Mock(), save_as_project_button=Mock(),
            webtoon_toggle=Mock(), progress_bar=Mock(),
            pipeline=SimpleNamespace(batch_process=Mock(), webtoon_batch_process=Mock(),
                prepare_translation_session=Mock(side_effect=lambda _paths: events.append('snapshot_request'))),
            default_error_handler=Mock(), on_batch_process_finished=Mock(),
            run_threaded=start_worker,
        )
        with patch.object(self.controller_module, "validate_settings", side_effect=lambda *_: events.append("validate") or True):
            self.controller_module.ComicTranslate._run_batch_for_paths(main, ["page.png"])
        return main, events, worker_snapshots

    def test_regular_pending_edit_is_saved_before_worker_snapshot(self):
        main, events, snapshots = self.run_startup()
        self.assertEqual(events, ["flush", "save_current", "validate", "snapshot_request", "worker"])
        self.assertEqual(snapshots[0]["page.png"]["viewer_state"]["text_items_state"], [{"text": "last pending edit"}])
        self.assertEqual(snapshots[0]["other.png"]["viewer_state"]["text_items_state"], [])
        main.pipeline.batch_process.assert_called_once_with(["page.png"])
        main.pipeline.webtoon_batch_process.assert_not_called()

    def test_webtoon_saves_all_live_pages_even_for_selected_page_batch(self):
        main, events, snapshots = self.run_startup(webtoon=True)
        self.assertEqual(events, ["flush", "save_all", "validate", "snapshot_request", "worker"])
        for path in main.image_files:
            self.assertEqual(snapshots[0][path]["viewer_state"]["text_items_state"], [{"text": "last pending edit " + path}])
        main.image_ctrl.save_current_image_state.assert_not_called()
        main.pipeline.webtoon_batch_process.assert_called_once_with(["page.png"])
        main.pipeline.batch_process.assert_not_called()

    def test_request_snapshot_preserves_pending_edits_even_without_auto_fit(self):
        main, events, snapshots = self.run_startup(enabled=False)
        self.assertEqual(events, ["flush", "save_current", "validate", "snapshot_request", "worker"])
        main.text_ctrl._commit_pending_text_command.assert_called_once()
        main.image_ctrl.save_current_image_state.assert_called_once()
        self.assertEqual(snapshots[0]["page.png"]["viewer_state"]["text_items_state"], [{"text": "last pending edit"}])

    def test_page_origin_uses_real_centering_converter(self):
        layout = SimpleNamespace(image_positions=[1000], webtoon_width=500)
        loader = SimpleNamespace(image_data={0: np.zeros((100, 200, 3), dtype=np.uint8)})
        converter = CoordinateConverter(layout, loader)
        viewer = SimpleNamespace(
            webtoon_mode=True,
            webtoon_manager=SimpleNamespace(coordinate_converter=converter),
            # The old layout convenience method omits horizontal centering.
            page_to_scene_coordinates=lambda _index, local: QPointF(local.x(), 1000 + local.y()),
        )
        controller = TextController.__new__(TextController)
        controller.main = SimpleNamespace(image_files=["page.png"], image_viewer=viewer)
        self.assertEqual(controller._render_page_origin("page.png"), QPointF(150, 1000))


if __name__ == "__main__":
    unittest.main()
