import hashlib
import os
import uuid

import imkit as imk
from PySide6.QtCore import QPointF
from PySide6.QtGui import QUndoCommand

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
    """Insert one inpainting pass as a fully undoable patch group."""

    def __init__(self, ct, patches, file_path, display=True):
        super().__init__("Inpaint")
        self.ct = ct
        self.viewer = ct.image_viewer
        self.scene = self.viewer._scene
        self.file_path = file_path
        self.display_hint = bool(display)
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
            self.properties_list.append(prop)

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
                if any(_same_patch_content(existing, prop) for existing in existing_content):
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
    """Remove every inpainting patch from one page, with undo restoration."""

    def __init__(self, ct, file_path: str):
        super().__init__("Revert inpainting")
        self.ct = ct
        self.viewer = ct.image_viewer
        self.scene = self.viewer._scene
        self.file_path = file_path
        self.properties_list = [
            dict(patch) for patch in ct.image_patches.get(file_path, [])
        ]

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

    def undo(self):
        patches = self.ct.image_patches.setdefault(self.file_path, [])
        for properties in self.properties_list:
            if not any(_same_patch_instance(existing, properties) for existing in patches):
                patches.append(dict(properties))

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

