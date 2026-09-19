import hashlib
import os
import uuid

import imkit as imk
from PySide6.QtCore import QCoreApplication, QPointF
from PySide6.QtGui import QUndoCommand

from app.projects.patch_metadata import (
    PATCH_KIND_FLUX2_EDIT,
    PATCH_KIND_INPAINT,
    normalize_patch_kind,
    patch_matches_kind,
    sanitize_patch_metadata,
)

from .base import PatchCommandBase


def _same_patch_content(left: dict, right: dict) -> bool:
    return left.get("hash") == right.get("hash") and list(left.get("bbox", [])) == list(
        right.get("bbox", [])
    )


def _same_patch_instance(left: dict, right: dict) -> bool:
    left_id = left.get("patch_id")
    right_id = right.get("patch_id")
    if left_id is not None and right_id is not None:
        return left_id == right_id
    return _same_patch_content(left, right)


def _page_is_visible(ct, file_path: str) -> bool:
    viewer = ct.image_viewer
    if viewer.webtoon_mode:
        if file_path not in ct.image_files:
            return False
        return ct.image_files.index(file_path) in viewer.webtoon_manager.loaded_pages
    if not viewer.hasPhoto() or not (0 <= ct.curr_img_idx < len(ct.image_files)):
        return False
    return ct.image_files[ct.curr_img_idx] == file_path


def _display_properties(ct, file_path: str, properties: dict) -> dict:
    prop = dict(properties)
    viewer = ct.image_viewer
    if viewer.webtoon_mode and file_path in ct.image_files:
        page_index = ct.image_files.index(file_path)
        x, y, _width, _height = prop["bbox"]
        scene_pos = viewer.page_to_scene_coordinates(page_index, QPointF(x, y))
        if scene_pos is not None:
            prop["scene_pos"] = [scene_pos.x(), scene_pos.y()]
            prop["page_index"] = page_index
    return prop


def _register_with_webtoon_patch_manager(viewer, item, prop: dict) -> None:
    if not viewer.webtoon_mode or item is None:
        return
    page_idx = prop.get("page_index")
    if page_idx is None:
        return
    manager = getattr(getattr(viewer, "webtoon_manager", None), "scene_item_manager", None)
    patch_manager = getattr(manager, "patch_manager", None) if manager is not None else None
    if patch_manager is None:
        return
    page_items = patch_manager.loaded_patch_items.setdefault(page_idx, [])
    if item not in page_items:
        page_items.append(item)

class PatchInsertCommand(QUndoCommand, PatchCommandBase):
    """Insert one patch group (inpaint pass or artistic edit) as an undo step."""

    def __init__(self, ct, patches, file_path, display=True, kind=None, text=None, metadata=None):
        # ``kind=None`` keeps records legacy-shaped (no kind field); callers
        # that opt in store an explicit kind on every generated patch record.
        self.kind = None if kind is None else normalize_patch_kind(kind)
        super().__init__(text or self._default_text(self.kind))
        self.ct = ct
        self.viewer = ct.image_viewer
        self.scene = self.viewer._scene
        self.file_path = file_path
        self.display_hint = bool(display)
        self.metadata = sanitize_patch_metadata(metadata)
        self._first_redo = True
        self._owned_patch_ids: set[str] | None = None

        self.properties_list = []
        for idx, patch in enumerate(patches):
            bbox = patch["bbox"]
            patch_img = patch["image"]
            patch_id = str(patch.get("patch_id") or uuid.uuid4().hex)

            sub_dir = os.path.join(
                ct.temp_dir, "inpaint_patches", os.path.basename(file_path)
            )
            os.makedirs(sub_dir, exist_ok=True)
            png_path = os.path.join(sub_dir, f"patch_{patch_id[:8]}_{idx}.png")
            imk.write_image(png_path, patch_img)

            with open(png_path, "rb") as file:
                image_bytes = file.read()
            image_hash = hashlib.sha256(
                image_bytes + str(bbox).encode("utf-8")
            ).hexdigest()

            prop = {
                "bbox": bbox,
                "png_path": png_path,
                "hash": image_hash,
                "patch_id": patch_id,
            }
            if "scene_pos" in patch:
                prop["scene_pos"] = patch["scene_pos"]
            if "page_index" in patch:
                prop["page_index"] = patch["page_index"]
            if self.kind is not None:
                prop["kind"] = self.kind
            if self.metadata:
                prop["metadata"] = dict(self.metadata)
            self.properties_list.append(prop)

    @staticmethod
    def _default_text(kind):
        if kind is None or kind == PATCH_KIND_INPAINT:
            return QCoreApplication.translate("PatchCommands", "Inpaint")
        if kind == PATCH_KIND_FLUX2_EDIT:
            return QCoreApplication.translate("PatchCommands", "Artistic edit")
        return QCoreApplication.translate("PatchCommands", "Patch edit")

    def _should_draw(self) -> bool:
        return _page_is_visible(self.ct, self.file_path) or (
            self._first_redo and self.display_hint
        )

    def _register_patches(self, load_into_memory: bool) -> None:
        patches_list = self.ct.image_patches.setdefault(self.file_path, [])
        if self._owned_patch_ids is None:
            self._owned_patch_ids = set()
            existing_content = list(patches_list)
            for prop in self.properties_list:
                # Instance identity (patch_id when available) keeps two
                # visually identical patches from deduplicating each other;
                # legacy records without ids still fall back to content.
                if any(_same_patch_instance(existing, prop) for existing in existing_content):
                    continue
                self._owned_patch_ids.add(prop["patch_id"])
                existing_content.append(prop)

        memory_list = self.ct.in_memory_patches.setdefault(self.file_path, [])
        for prop in self.properties_list:
            if prop["patch_id"] not in self._owned_patch_ids:
                continue
            if not any(_same_patch_instance(existing, prop) for existing in patches_list):
                patches_list.append(dict(prop))
            if load_into_memory and not any(
                _same_patch_instance(existing, prop) for existing in memory_list
            ):
                image = imk.read_image(prop["png_path"])
                if image is not None:
                    memory_list.append(
                        {
                            "bbox": prop["bbox"],
                            "image": image,
                            "hash": prop["hash"],
                            "patch_id": prop["patch_id"],
                        }
                    )

    def _unregister_patches(self) -> None:
        owned = self._owned_patch_ids or set()
        patches_list = self.ct.image_patches.get(self.file_path, [])
        patches_list[:] = [
            patch for patch in patches_list if patch.get("patch_id") not in owned
        ]
        memory_list = self.ct.in_memory_patches.get(self.file_path, [])
        memory_list[:] = [
            patch for patch in memory_list if patch.get("patch_id") not in owned
        ]

    def _draw_pixmaps(self) -> None:
        if not self._should_draw():
            return
        owned = self._owned_patch_ids or set()
        for properties in self.properties_list:
            if properties["patch_id"] not in owned:
                continue
            prop = _display_properties(self.ct, self.file_path, properties)
            item = self.find_matching_item(self.scene, prop)
            if item is None:
                item = self.create_patch_item(prop, self.viewer)
            _register_with_webtoon_patch_manager(self.viewer, item, prop)

    def _remove_pixmaps(self) -> None:
        owned = self._owned_patch_ids or set()
        for properties in self.properties_list:
            if properties["patch_id"] not in owned:
                continue
            prop = _display_properties(self.ct, self.file_path, properties)
            item = self.find_matching_item(self.scene, prop)
            if item is not None:
                self.remove_patch_item(self.viewer, item)

    def redo(self):
        should_draw = self._should_draw()
        self._register_patches(load_into_memory=should_draw)
        self._draw_pixmaps()
        self._first_redo = False

    def undo(self):
        self._remove_pixmaps()
        self._unregister_patches()


class PatchClearCommand(QUndoCommand, PatchCommandBase):
    """Remove one kind of patch from a page, with undo restoration.

    ``kind=None`` keeps the historical "revert everything" behavior. The
    Revert Inpainting action passes ``kind="inpaint"`` so accepted artistic
    edits are never removed by it (legacy patches without a kind count as
    inpaint).
    """

    def __init__(self, ct, file_path: str, kind=None, text=None):
        self.kind = None if kind is None else normalize_patch_kind(kind)
        if text is None:
            if self.kind is None:
                text = QCoreApplication.translate("PatchCommands", "Revert all patches")
            elif self.kind == PATCH_KIND_INPAINT:
                text = QCoreApplication.translate("PatchCommands", "Revert inpainting")
            elif self.kind == PATCH_KIND_FLUX2_EDIT:
                text = QCoreApplication.translate("PatchCommands", "Revert artistic edits")
            else:
                text = QCoreApplication.translate("PatchCommands", "Revert patches")
        super().__init__(text)
        self.ct = ct
        self.viewer = ct.image_viewer
        self.scene = self.viewer._scene
        self.file_path = file_path
        # Remember each captured patch's original list position so undo can
        # restore the canonical compositing order relative to patches of
        # other kinds that stayed in the list.
        self.captured = [
            (index, dict(patch))
            for index, patch in enumerate(ct.image_patches.get(file_path, []))
            if self.kind is None or patch_matches_kind(patch, self.kind)
        ]
        self.properties_list = [properties for _index, properties in self.captured]

    def redo(self):
        for properties in self.properties_list:
            prop = _display_properties(self.ct, self.file_path, properties)
            item = self.find_matching_item(self.scene, prop)
            if item is not None:
                self.remove_patch_item(self.viewer, item)

        captured = self.properties_list
        patches = self.ct.image_patches.get(self.file_path, [])
        patches[:] = [
            patch
            for patch in patches
            if not any(_same_patch_instance(patch, saved) for saved in captured)
        ]
        memory = self.ct.in_memory_patches.get(self.file_path, [])
        memory[:] = [
            patch
            for patch in memory
            if not any(_same_patch_instance(patch, saved) for saved in captured)
        ]

    def _restore_visual_order(self) -> None:
        """Match equal-Z scene stacking to the persisted patch-list order."""
        ordered_items = []
        for properties in self.ct.image_patches.get(self.file_path, []):
            prop = _display_properties(self.ct, self.file_path, properties)
            item = self.find_matching_item(self.scene, prop)
            if item is not None:
                ordered_items.append(item)

        # Patch items intentionally share z=0.5. Qt otherwise places a newly
        # restored item above its untouched siblings, even when its record was
        # reinserted at an earlier list position. Re-add the page's items in
        # canonical bottom-to-top order. Keep Python references throughout so
        # removing an item never releases it before it is returned to the scene.
        for item in ordered_items:
            self.scene.removeItem(item)
        for item in ordered_items:
            self.scene.addItem(item)
        self.scene.update()

    def undo(self):
        patches = self.ct.image_patches.setdefault(self.file_path, [])
        captured_props = [properties for _index, properties in self.captured]
        restored = [
            patch
            for patch in patches
            if not any(_same_patch_instance(patch, saved) for saved in captured_props)
        ]
        # Re-insert at the original indices (ascending) so a filtered revert
        # restores the exact pre-clear order around the untouched kinds.
        for index, properties in self.captured:
            restored.insert(min(index, len(restored)), dict(properties))
        patches[:] = restored

        if not _page_is_visible(self.ct, self.file_path):
            return

        memory = self.ct.in_memory_patches.setdefault(self.file_path, [])
        for properties in self.properties_list:
            image = imk.read_image(properties["png_path"])
            if image is not None and not any(
                _same_patch_instance(existing, properties) for existing in memory
            ):
                memory_entry = {
                    "bbox": properties["bbox"],
                    "image": image,
                    "hash": properties["hash"],
                }
                if properties.get("patch_id") is not None:
                    memory_entry["patch_id"] = properties["patch_id"]
                memory.append(memory_entry)

            prop = _display_properties(self.ct, self.file_path, properties)
            item = self.find_matching_item(self.scene, prop)
            if item is None:
                item = self.create_patch_item(prop, self.viewer)
            _register_with_webtoon_patch_manager(self.viewer, item, prop)

        self._restore_visual_order()
