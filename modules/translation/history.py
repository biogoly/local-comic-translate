"""Immutable, project-local readings that still match a translation review."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json

from modules.review.issues import fingerprint_for_stage
from .context import canonical_language


@dataclass(frozen=True)
class ApprovedExample:
    page_id: str
    region_id: str
    source_lang: str
    target_lang: str
    source: str
    target: str

    def reference(self):
        return {'source': self.source, 'translation': self.target}


@dataclass(frozen=True)
class ApprovedHistory:
    page_ids: tuple[str, ...]
    examples: tuple[ApprovedExample, ...]

    def for_pair(self, source_lang, target_lang, *, before_page_ids=()):
        source, target = canonical_language(source_lang), canonical_language(target_lang)
        boundaries = tuple(before_page_ids)
        if any(not p or self.page_ids.count(p) != 1 for p in boundaries):
            return ()
        end = min((self.page_ids.index(p) for p in boundaries), default=len(self.page_ids))
        eligible = set(self.page_ids[:end])
        return tuple(row for row in self.examples if row.page_id in eligible
                     and row.source_lang == source and row.target_lang == target)

    def select(self, source, target, pages, excluded, byte_budget):
        """Choose recent complete pairs and retain chronological prompt order."""
        if not pages or byte_budget <= 0:
            return ()
        seen, selected = set(), []
        for row in reversed(self.for_pair(source, target, before_page_ids=pages)):
            pair = row.source, row.target
            if row.region_id in excluded or pair in seen:
                continue
            seen.add(pair)
            # JSON punctuation, escape sequences and UTF-8 all count.
            size = len(json.dumps(row.reference(), ensure_ascii=False, sort_keys=True,
                                  separators=(',', ':')).encode('utf-8')) + bool(selected)
            if size <= byte_budget:
                selected.append(row)
                byte_budget -= size
                if len(selected) == 24:
                    break
        return tuple(reversed(selected))


@dataclass(frozen=True)
class HistoryGuard:
    source: str
    target: str
    pages: tuple
    excluded: frozenset
    byte_budget: int
    included: tuple[ApprovedExample, ...]

    def matches(self, snapshot):
        return snapshot.select(self.source, self.target, self.pages, self.excluded, self.byte_budget) == self.included

    def current(self, main):
        return self.matches(capture_approved_history(main))


def history_is_current(request, main):
    guard = getattr(request, 'history_guard', None)
    return not isinstance(guard, HistoryGuard) or guard.current(main)


def capture_approved_history(main):
    """Read only block text/approval data. Loaded pages supersede saved mirrors."""
    paths = tuple(getattr(main, 'image_files', ()))
    states = dict(getattr(main, 'image_states', {}))
    ids = tuple(states.get(path, {}).get('page_id', '') for path in paths)
    page_counts = Counter(ids)
    blocks = [tuple(states.get(path, {}).get('blk_list', ())) for path in paths]
    if hasattr(main, 'blk_list'):
        if getattr(main, 'webtoon_mode', False):
            manager = getattr(getattr(main, 'image_viewer', None), 'webtoon_manager', None)
            loaded = {i for i in getattr(manager, 'loaded_pages', ()) if 0 <= i < len(paths)}
            owners = {}
            for i in loaded:
                for block in blocks[i]:
                    owners.setdefault(block.region_id, set()).add(i)
            live = {i: [] for i in loaded}
            for block in tuple(main.blk_list):
                page = getattr(block, '_page_id', '')
                if page and page_counts[page] == 1:
                    index = ids.index(page)
                else:
                    known = owners.get(block.region_id, ())
                    if len(known) == 1:
                        index = next(iter(known))
                    else:
                        # Unsaved new regions and clipped webtoon fragments have
                        # scene coordinates; use the physical page's vertical extent.
                        y = (float(block.xyxy[1]) + float(block.xyxy[3])) / 2
                        positions, heights = getattr(manager, 'image_positions', ()), getattr(manager, 'image_heights', ())
                        matches = [i for i in loaded if i < len(positions) and i < len(heights)
                                   and positions[i] <= y < positions[i] + heights[i]]
                        index = matches[0] if len(matches) == 1 else None
                if index in live:
                    live[index].append(block)
            for index, current in live.items():
                blocks[index] = current
        else:
            index = getattr(main, 'curr_img_idx', -1)
            if 0 <= index < len(paths):
                blocks[index] = tuple(main.blk_list)

    rows = []
    mapping = getattr(main, 'lang_mapping', {})
    for path, page, group in zip(paths, ids, blocks):
        if not page or page_counts[page] != 1:
            continue
        state = states[path]
        counts = Counter(block.region_id for block in group)
        for block in group:
            identity = block.region_id
            if not identity or counts[identity] != 1:
                continue
            approval = getattr(block, 'region_metadata', {}).get('review', {}).get('approved_fingerprints', {}).get('translation')
            if not approval or not block.text or not block.text.strip() or not block.translation or not block.translation.strip():
                continue
            languages = {'source': state.get('source_lang', getattr(block, 'source_lang', '')),
                         'target': state.get('target_lang', getattr(block, 'target_lang', ''))}
            if approval != fingerprint_for_stage('translation', block, languages=languages):
                continue
            source, target = (canonical_language(mapping.get(s, s)) for s in languages.values())
            # A detected script or Auto setting alone is not proof of a language.
            if source in ('', 'auto') or target in ('', 'auto'):
                continue
            rows.append(ApprovedExample(page, identity, source, target, block.text, block.translation))
    return ApprovedHistory(ids, tuple(rows))
