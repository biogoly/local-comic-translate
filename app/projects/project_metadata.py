"""Backward-compatible page IDs and conservative legacy item linkage."""
from __future__ import annotations

import copy
from collections import Counter

from modules.utils.region_metadata import (
    ensure_region_id, find_region, new_id, normalized_id, normalize_layout_preferences,
    normalize_region_metadata,
)


def normalize_page_state(state: dict) -> dict:
    """Normalize metadata in place without changing any pixels, text or geometry."""
    state["page_id"] = normalized_id(state.get("page_id")) or new_id()
    blocks = state.get("blk_list", []) or []
    for block in blocks:
        ensure_region_id(block)
        block.region_metadata = normalize_region_metadata(getattr(block, "region_metadata", None))
    items = state.get("viewer_state", {}).get("text_items_state", []) or []
    claimed = {normalized_id(item.get("region_id")) for item in items if item.get("region_id")}
    candidates = []
    for item in items:
        item["layout_preferences"] = normalize_layout_preferences(item.get("layout_preferences"))
        if item.get("region_id"):
            # Preserve explicit stale IDs. They must not fall through to geometry.
            continue
        position = item.get("position", ())
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            continue
        block = find_region(blocks, x=position[0], y=position[1], angle=item.get("rotation", 0))
        if block is not None and block.region_id not in claimed:
            candidates.append((item, block.region_id))
    counts = Counter(identity for _, identity in candidates)
    for item, identity in candidates:
        if counts[identity] == 1:
            item["region_id"] = identity
    return state


def duplicate_page_state(state: dict) -> dict:
    """An explicit copied page gets new page and region identities."""
    duplicate = copy.deepcopy(state)
    duplicate["page_id"] = new_id()
    mapping = {}
    for block in duplicate.get("blk_list", []) or []:
        old = ensure_region_id(block)
        mapping.setdefault(old, new_id())
        block.region_id = mapping[old]
        block.region_metadata = normalize_region_metadata({
            "layout_preferences": getattr(block, "region_metadata", {}).get("layout_preferences", {})
        })
    viewer_state = duplicate.get("viewer_state", {})
    for collection in ("text_items_state", "rectangles"):
        for item in viewer_state.get(collection, []) or []:
            old = item.get("region_id")
            if old:
                mapping.setdefault(old, new_id())
                item["region_id"] = mapping[old]
    return normalize_page_state(duplicate)


def normalize_project_pages(controller) -> None:
    """Run before writing or after path remapping; order and filenames are irrelevant."""
    states = controller.image_states
    seen = set()
    paths = list(dict.fromkeys(list(controller.image_files) + list(states)))
    for path in paths:
        state = states.setdefault(path, {})
        if not isinstance(state, dict):
            state = states[path] = {}
        normalize_page_state(state)
        if state["page_id"] in seen:
            state = states[path] = duplicate_page_state(state)
        seen.add(state["page_id"])
