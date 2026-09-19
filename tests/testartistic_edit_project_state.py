import hashlib
import os
import shutil
import tempfile
import unittest
import zipfile
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import imkit as imk
import msgpack
import numpy as np

from app.path_materialization import ensure_path_materialized
from app.projects.patch_metadata import (
    MAX_METADATA_KEY_LENGTH,
    MAX_METADATA_STRING_LENGTH,
    normalize_patch_kind,
    sanitize_patch_metadata,
)
from app.projects.project_state import load_state_from_proj_file
from app.projects.project_state_v2 import (
    close_cached_connection,
    load_state_from_proj_file_v2,
    save_state_to_proj_file_v2,
)


def _write_png(path, color):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :] = color
    imk.write_image(path, image)
    return path


def _patch_hash(png_path, bbox):
    with open(png_path, "rb") as file:
        payload = file.read()
    return hashlib.sha256(payload + str(bbox).encode("utf-8")).hexdigest()


class _StubSettings:
    def get_llm_settings(self):
        return {"extra_context": ""}


def _make_controller(temp_dir, image_files=None, image_patches=None):
    return SimpleNamespace(
        temp_dir=temp_dir,
        image_files=list(image_files or []),
        image_data={},
        in_memory_history={},
        image_history={},
        image_states={},
        image_patches=dict(image_patches or {}),
        current_history_index={},
        displayed_images=set(),
        loaded_images=[],
        curr_img_idx=0,
        webtoon_mode=False,
        image_viewer=SimpleNamespace(webtoon_view_state={}),
        settings_page=_StubSettings(),
    )


FLUX_METADATA = {
    "prompt": "Replace the title lettering",
    "model_key": "klein-4b",
    "model_source": "black-forest-labs/FLUX.2-klein-4B",
    "seed": 123,
    "steps": 4,
    "guidance": 1.0,
    "lora_name": "manga.safetensors",
    "lora_sha256": "abc123",
    "lora_scale": 0.8,
    "selection_mode": "selected_box",
    "source_crop_sha256": "def456",
    "working_scale": 1.0,
    "future_key": [1, 2, 3],
}


class ProjectStateV2TypedPatchTests(unittest.TestCase):
    """Mixed legacy/inpaint/FLUX patches survive a v2 save/load cycle."""

    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.root = self.workspace.name
        self.save_dir = os.path.join(self.root, "save")
        self.load_dir = os.path.join(self.root, "load")
        os.makedirs(self.save_dir)
        os.makedirs(self.load_dir)
        self.project_file = os.path.join(self.root, "project.ctpr")

        self.page = _write_png(os.path.join(self.save_dir, "page.png"), (10, 20, 30))

        def make_patch(name, bbox, color, **extra):
            png_path = _write_png(os.path.join(self.save_dir, name), color)
            record = {
                "bbox": list(bbox),
                "png_path": png_path,
                "hash": _patch_hash(png_path, list(bbox)),
            }
            record.update(extra)
            return record

        self.legacy_patch = make_patch("legacy.png", [1, 1, 4, 4], (255, 0, 0))
        self.inpaint_patch = make_patch(
            "inpaint.png",
            [2, 2, 4, 4],
            (0, 255, 0),
            patch_id="inpaint-1",
            kind="inpaint",
        )
        self.flux_patch = make_patch(
            "flux.png",
            [3, 3, 4, 4],
            (0, 0, 255),
            patch_id="flux-1",
            kind="flux2_edit",
            metadata=dict(FLUX_METADATA),
        )

    def tearDown(self):
        close_cached_connection(self.project_file)
        self.workspace.cleanup()

    def _save_and_reload(self):
        controller = _make_controller(
            self.save_dir,
            [self.page],
            {self.page: [self.legacy_patch, self.inpaint_patch, self.flux_patch]},
        )
        save_state_to_proj_file_v2(controller, self.project_file)

        loaded = _make_controller(self.load_dir)
        load_state_from_proj_file_v2(loaded, self.project_file)
        self.assertEqual(len(loaded.image_files), 1)
        loaded_page = loaded.image_files[0]
        self.assertIn(loaded_page, loaded.image_patches)
        return loaded.image_patches[loaded_page]

    def test_order_kind_and_metadata_survive_reload(self):
        patches = self._save_and_reload()

        self.assertEqual(
            [patch["bbox"] for patch in patches],
            [[1, 1, 4, 4], [2, 2, 4, 4], [3, 3, 4, 4]],
        )
        # Legacy patches stay kind-less and behave as inpaint.
        self.assertNotIn("kind", patches[0])
        self.assertNotIn("metadata", patches[0])

        self.assertEqual(patches[1]["kind"], "inpaint")
        self.assertEqual(patches[1]["patch_id"], "inpaint-1")

        self.assertEqual(patches[2]["kind"], "flux2_edit")
        self.assertEqual(patches[2]["patch_id"], "flux-1")
        self.assertEqual(patches[2]["metadata"], FLUX_METADATA)

    def test_patch_hashes_survive_reload(self):
        patches = self._save_and_reload()
        self.assertEqual(
            [patch["hash"] for patch in patches],
            [self.legacy_patch["hash"], self.inpaint_patch["hash"], self.flux_patch["hash"]],
        )

    def test_png_blobs_materialize_lazily_and_match_original_bytes(self):
        patches = self._save_and_reload()
        originals = [self.legacy_patch, self.inpaint_patch, self.flux_patch]

        for patch_entry, original in zip(patches, originals):
            self.assertFalse(os.path.isfile(patch_entry["png_path"]))
            self.assertTrue(ensure_path_materialized(patch_entry["png_path"]))
            with open(patch_entry["png_path"], "rb") as loaded_file:
                loaded_bytes = loaded_file.read()
            with open(original["png_path"], "rb") as original_file:
                self.assertEqual(loaded_bytes, original_file.read())

    def test_existing_bbox_only_projects_still_load(self):
        # A project whose patch records only carry bbox/png/hash must load
        # unchanged, with no synthetic kind/metadata fields.
        controller = _make_controller(
            self.save_dir, [self.page], {self.page: [self.legacy_patch]}
        )
        save_state_to_proj_file_v2(controller, self.project_file)

        loaded = _make_controller(self.load_dir)
        load_state_from_proj_file_v2(loaded, self.project_file)
        loaded_page = loaded.image_files[0]
        patches = loaded.image_patches[loaded_page]
        self.assertEqual(len(patches), 1)
        self.assertNotIn("kind", patches[0])
        self.assertNotIn("metadata", patches[0])
        self.assertNotIn("patch_id", patches[0])


class LegacyProjectTypedPatchTests(unittest.TestCase):
    """The legacy ZIP/msgpack loader must pass kind/metadata through."""

    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.root = self.workspace.name
        self.ctpr_path = os.path.join(self.root, "legacy.ctpr")
        self.original_page = "C:\\comics\\page1.png"

        staging = os.path.join(self.root, "staging")
        unique_images = os.path.join(staging, "unique_images")
        unique_patches = os.path.join(staging, "unique_patches")
        os.makedirs(unique_images)
        os.makedirs(unique_patches)
        _write_png(os.path.join(unique_images, "0.png"), (9, 9, 9))
        _write_png(os.path.join(unique_patches, "legacy.png"), (200, 0, 0))
        _write_png(os.path.join(unique_patches, "flux.png"), (0, 0, 200))

        state = {
            "current_image_index": 0,
            "original_image_files": [self.original_page],
            "image_files_references": {self.original_page: "0"},
            "unique_images": {"0": "0.png"},
            "image_patches": {
                self.original_page: [
                    {
                        "bbox": [0, 0, 4, 4],
                        "png_path": "legacy.png",
                        "hash": "legacy-hash",
                    },
                    {
                        "bbox": [2, 2, 4, 4],
                        "png_path": "flux.png",
                        "hash": "flux-hash",
                        "patch_id": "legacy-flux-1",
                        "kind": "flux2_edit",
                        "metadata": {
                            "prompt": "keep me",
                            "seed": 5,
                            "junk": {"nested": [1]},
                            "not_a_number": float("nan"),
                        },
                    },
                ]
            },
        }
        with open(os.path.join(staging, "state.msgpack"), "wb") as file:
            file.write(msgpack.packb(state, use_bin_type=True))
        with zipfile.ZipFile(self.ctpr_path, "w") as archive:
            for name in (
                "state.msgpack",
                os.path.join("unique_images", "0.png"),
                os.path.join("unique_patches", "legacy.png"),
                os.path.join("unique_patches", "flux.png"),
            ):
                archive.write(os.path.join(staging, name), name)
        shutil.rmtree(staging, ignore_errors=True)

    def tearDown(self):
        self.workspace.cleanup()

    def test_legacy_load_treats_untyped_patches_as_inpaint(self):
        loaded = _make_controller(os.path.join(self.root, "load"))
        load_state_from_proj_file(loaded, self.ctpr_path)

        patches = loaded.image_patches[loaded.image_files[0]]
        self.assertEqual(len(patches), 2)

        legacy = patches[0]
        self.assertNotIn("kind", legacy)
        self.assertEqual(legacy["hash"], "legacy-hash")
        self.assertTrue(os.path.isfile(legacy["png_path"]))

    def test_legacy_load_carries_kind_and_sanitized_metadata(self):
        loaded = _make_controller(os.path.join(self.root, "load"))
        load_state_from_proj_file(loaded, self.ctpr_path)

        flux = loaded.image_patches[loaded.image_files[0]][1]
        self.assertEqual(flux["kind"], "flux2_edit")
        self.assertEqual(flux["patch_id"], "legacy-flux-1")
        self.assertEqual(flux["metadata"], {"prompt": "keep me", "seed": 5})


class PatchMetadataSanitizerTests(unittest.TestCase):
    def test_non_mapping_returns_empty(self):
        self.assertEqual(sanitize_patch_metadata(None), {})
        self.assertEqual(sanitize_patch_metadata(["prompt"]), {})
        self.assertEqual(sanitize_patch_metadata("prompt"), {})

    def test_primitive_values_survive(self):
        sanitized = sanitize_patch_metadata(
            {
                "text": "value",
                "count": 12,
                "ratio": 0.75,
                "flag": False,
                "none": None,
                "words": ["a", "b", 3, 0.5, True],
                "future_unknown": 42,
            }
        )
        self.assertEqual(
            sanitized,
            {
                "text": "value",
                "count": 12,
                "ratio": 0.75,
                "flag": False,
                "none": None,
                "words": ["a", "b", 3, 0.5, True],
                "future_unknown": 42,
            },
        )

    def test_unsafe_values_are_dropped(self):
        sanitized = sanitize_patch_metadata(
            {
                "nested": {"a": 1},
                "nan": float("nan"),
                "infinity": float("inf"),
                "huge_int": 2 ** 64,
                "raw": b"binary",
                7: "non-string key",
                "": "empty key",
            }
        )
        self.assertEqual(sanitized, {})

    def test_long_strings_and_keys_are_truncated(self):
        sanitized = sanitize_patch_metadata(
            {"k" * 200: "v" * (MAX_METADATA_STRING_LENGTH + 500)}
        )
        keys = list(sanitized)
        self.assertEqual(len(keys), 1)
        self.assertEqual(len(keys[0]), MAX_METADATA_KEY_LENGTH)
        self.assertEqual(len(sanitized[keys[0]]), MAX_METADATA_STRING_LENGTH)

    def test_list_entries_are_primitives_only(self):
        sanitized = sanitize_patch_metadata(
            {"mixed": [1, {"drop": True}, "keep", float("nan"), b"bin"]}
        )
        self.assertEqual(sanitized, {"mixed": [1, "keep"]})

    def test_normalize_patch_kind(self):
        self.assertEqual(normalize_patch_kind(None), "inpaint")
        self.assertEqual(normalize_patch_kind("flux2_edit"), "flux2_edit")


if __name__ == "__main__":
    unittest.main()
