"""Identity and fit metadata must survive movement, copies and project storage."""
import copy
import os
import tempfile
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import imkit as imk
import msgpack
import numpy as np
from PySide6.QtWidgets import QApplication, QGraphicsScene
from PySide6.QtCore import QPointF

from app.projects.parsers import ProjectDecoder, ProjectEncoder
from app.projects.project_metadata import normalize_page_state, normalize_project_pages, duplicate_page_state
from app.projects.project_state_v2 import save_state_to_proj_file_v2, load_state_from_proj_file_v2, close_cached_connection
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import TextBlockItem
from app.ui.commands.base import RectCommandBase
from modules.utils.region_metadata import find_region, normalized_id, normalize_layout_preferences
from modules.utils.textblock import TextBlock
from app.ui.canvas.webtoons.scene_items.text_block_manager import TextBlockManager
from app.ui.canvas.webtoons.scene_items.text_item_manager import TextItemManager


def block(coords=(10, 20, 110, 120), **kwargs):
    return TextBlock(text_bbox=np.array(coords), **kwargs)


def item_state(**kwargs):
    return {"position": (10, 20), "rotation": 0, "text": "Bonjour", **kwargs}


class RegionIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_snapshot_keeps_identity_but_duplicate_does_not(self):
        original = block(region_metadata={"protections": {"keep_translation": True}})
        snapshot = original.deep_copy()
        snapshot.xyxy[0] = 90
        snapshot.region_metadata["protections"]["keep_translation"] = False
        self.assertEqual(snapshot.region_id, original.region_id)
        self.assertEqual(original.xyxy[0], 10)
        self.assertTrue(original.region_metadata["protections"]["keep_translation"])
        duplicate = original.duplicate()
        self.assertNotEqual(duplicate.region_id, original.region_id)
        self.assertFalse(duplicate.region_metadata["protections"]["keep_translation"])

    def test_legacy_block_gets_id_once_and_roundtrips(self):
        legacy = block()
        del legacy.region_id
        del legacy.region_metadata
        encoded = ProjectEncoder.encode_textblock(legacy)
        identity = legacy.region_id
        restored = ProjectDecoder.decode_textblock(encoded)
        self.assertEqual(restored.region_id, identity)
        self.assertEqual(legacy.deep_copy().region_id, identity)
        packed = msgpack.packb(restored, default=ProjectEncoder().encode, use_bin_type=True)
        unpacked = msgpack.unpackb(packed, object_hook=ProjectDecoder().decode)
        self.assertEqual(unpacked.region_id, identity)
        self.assertEqual(normalized_id(identity), identity)

    def test_explicit_identity_does_not_fall_back_to_nearby_box(self):
        original = block()
        neighbour = block()
        self.assertIsNone(find_region([neighbour], region_id=original.region_id, x=10, y=20))
        original.xyxy = np.array([300, 400, 500, 600])
        self.assertIs(find_region([original, neighbour], region_id=original.region_id), original)
        self.assertIsNone(find_region([block(), block()], x=10, y=20))

    def test_legacy_linkage_is_unique_in_both_directions(self):
        region = block()
        state = {"blk_list": [region], "viewer_state": {"text_items_state": [item_state()]}}
        normalize_page_state(state)
        self.assertEqual(state["viewer_state"]["text_items_state"][0]["region_id"], region.region_id)
        ambiguous = {"blk_list": [block(), block()], "viewer_state": {"text_items_state": [item_state()]}}
        normalize_page_state(ambiguous)
        self.assertNotIn("region_id", ambiguous["viewer_state"]["text_items_state"][0])
        two_items = {"blk_list": [block()], "viewer_state": {"text_items_state": [item_state(), item_state()]}}
        normalize_page_state(two_items)
        self.assertTrue(all("region_id" not in item for item in two_items["viewer_state"]["text_items_state"]))

    def test_page_duplication_relinks_items_and_does_not_mutate_source(self):
        region = block()
        state = normalize_page_state({"blk_list": [region], "viewer_state": {"text_items_state": [item_state()]}})
        duplicate = duplicate_page_state(state)
        self.assertNotEqual(duplicate["page_id"], state["page_id"])
        new_region = duplicate["blk_list"][0]
        self.assertNotEqual(new_region.region_id, region.region_id)
        self.assertEqual(duplicate["viewer_state"]["text_items_state"][0]["region_id"], new_region.region_id)
        self.assertEqual(state["viewer_state"]["text_items_state"][0]["region_id"], region.region_id)

    def test_reordering_keeps_page_ids_and_duplicate_state_gets_a_new_id(self):
        controller = SimpleNamespace(image_files=["a", "b"], image_states={"a": {}, "b": {}})
        normalize_project_pages(controller)
        identities = {p: state["page_id"] for p, state in controller.image_states.items()}
        controller.image_files.reverse()
        normalize_project_pages(controller)
        self.assertEqual(identities, {p: state["page_id"] for p, state in controller.image_states.items()})
        controller.image_states["c"] = copy.deepcopy(controller.image_states["a"])
        controller.image_files.append("c")
        normalize_project_pages(controller)
        self.assertNotEqual(controller.image_states["a"]["page_id"], controller.image_states["c"]["page_id"])

    def test_item_fit_preferences_roundtrip_without_aliasing(self):
        region = block()
        item = TextBlockItem()
        item.region_id = region.region_id
        item.layout_preferences = {"bound_width": 100, "bound_height": 80,
                                   "min_font_size": 11, "keep_manual_breaks": True}
        props = TextItemProperties.from_text_item(item)
        restored = TextItemProperties.from_dict(props.to_dict())
        self.assertEqual(restored.region_id, region.region_id)
        self.assertEqual(restored.layout_preferences, item.layout_preferences)
        self.assertEqual(restored.document_font, item.document().defaultFont().toString())
        restored.layout_preferences["bound_width"] = 200
        self.assertEqual(item.layout_preferences["bound_width"], 100)
        self.assertEqual(TextItemProperties.from_dict({}).layout_preferences, {})
        self.assertEqual(normalize_layout_preferences({"bound_width": float("nan"), "bound_height": -5}), {})

    def test_undo_lookup_follows_identity_after_movement(self):
        region = block()
        props = copy.deepcopy(region.__dict__)
        region.xyxy = np.array([90, 80, 190, 180])
        self.assertIs(RectCommandBase.find_matching_blk([region], props), region)
        self.assertIsNone(RectCommandBase.find_matching_blk([block()], props))
        scene = QGraphicsScene()
        item = TextBlockItem()
        item.region_id = region.region_id
        scene.addItem(item)
        item_props = TextItemProperties.from_text_item(item)
        item.setPos(200, 300)
        self.assertIs(RectCommandBase.find_matching_txt_item(scene, item_props), item)

    def test_webtoon_coordinate_and_clipped_copies_retain_identity(self):
        region = block()
        converter = SimpleNamespace(
            page_local_to_scene_position=lambda point, page: point + QPointF(0, 800),
            scene_to_page_local_position=lambda point, page: point - QPointF(0, 800),
        )
        manager = TextBlockManager(SimpleNamespace(_scene=None), None, converter, None)
        manager._convert_textblock_coordinates(region, 1, to_scene=True)
        self.assertEqual(region.xyxy[1], 820)
        clipped = manager._create_clipped_textblock(region, (10, 0, 110, 60), 1)
        self.assertEqual(clipped.region_id, region.region_id)
        manager._convert_textblock_coordinates(region, 1, to_scene=False)
        np.testing.assert_array_equal(region.xyxy, [10, 20, 110, 120])
        props = item_state(region_id=region.region_id, layout_preferences={"bound_width": 100})
        text_manager = TextItemManager(SimpleNamespace(_scene=None), None, converter, None)
        clipped_props = text_manager._create_clipped_text_data(props, 'Bonjour', (10, 0), (100, 60))
        self.assertEqual(clipped_props['region_id'], region.region_id)
        self.assertEqual(clipped_props['layout_preferences'], props['layout_preferences'])

    def test_undo_lookup_distinguishes_fragments_of_one_logical_region(self):
        first = block()
        second = first.deep_copy()
        second.xyxy = np.array([10, 820, 110, 920])
        self.assertIs(RectCommandBase.find_matching_blk([first, second], second.__dict__), second)
        scene = QGraphicsScene()
        items = [TextBlockItem(), TextBlockItem()]
        for index, item in enumerate(items):
            item.region_id = first.region_id
            item.setPos(10, 20 + index * 800)
            scene.addItem(item)
        props = TextItemProperties.from_text_item(items[1])
        self.assertIs(RectCommandBase.find_matching_txt_item(scene, props), items[1])


def controller(directory, pages=()):
    return SimpleNamespace(
        temp_dir=directory, image_files=list(pages), image_data={}, in_memory_history={},
        image_history={}, image_states={}, image_patches={}, current_history_index={},
        displayed_images=set(), loaded_images=[], curr_img_idx=0, webtoon_mode=False,
        image_viewer=SimpleNamespace(webtoon_view_state={}),
        settings_page=SimpleNamespace(get_llm_settings=lambda: {"extra_context": ""}),
    )


class IdentityProjectRoundtripTests(unittest.TestCase):
    def test_sqlite_save_as_path_remap_and_reopen_keep_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "page.png")
            imk.write_image(source, np.full((20, 30, 3), 240, dtype=np.uint8))
            first = controller(directory, [source])
            region = block(text="Hello", translation="Bonjour", is_manual=True)
            first.image_states[source] = {"blk_list": [region], "viewer_state": {"text_items_state": [item_state()]}}
            project = os.path.join(directory, "first.ctpr")
            save_as = os.path.join(directory, "copy.ctpr")
            try:
                save_state_to_proj_file_v2(first, project)
                page_id = first.image_states[source]["page_id"]
                load_dir = os.path.join(directory, "loaded")
                os.makedirs(load_dir)
                loaded = controller(load_dir)
                load_state_from_proj_file_v2(loaded, project)
                restored = loaded.image_states[loaded.image_files[0]]
                self.assertEqual(restored["page_id"], page_id)
                self.assertEqual(restored["blk_list"][0].region_id, region.region_id)
                self.assertEqual(restored["viewer_state"]["text_items_state"][0]["region_id"], region.region_id)
                save_state_to_proj_file_v2(loaded, save_as)
                other_dir = os.path.join(directory, "reopened")
                os.makedirs(other_dir)
                reopened = controller(other_dir)
                load_state_from_proj_file_v2(reopened, save_as)
                self.assertEqual(reopened.image_states[reopened.image_files[0]]["page_id"], page_id)
            finally:
                close_cached_connection(project)
                close_cached_connection(save_as)


if __name__ == "__main__":
    unittest.main()
