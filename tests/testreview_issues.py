"""Pure issue derivation, content-bound approvals and bounded persisted metadata."""
import copy
from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

from modules.review.issues import (
    ReviewIssue, clear_review, dismiss_issue, evaluate_region,
    fingerprint_for_stage, mark_reviewed,
)
from modules.ocr.evidence import make_ocr_evidence
from modules.utils.region_metadata import (
    MAX_DISMISSED_FINGERPRINTS, normalize_region_metadata, normalize_review_metadata,
)


def region(**changes):
    values = dict(region_id="region-1", text="Bonjour", translation="Hello",
                  xyxy=[10, 20, 100, 90], bubble_xyxy=[0, 0, 120, 100], angle=0,
                  direction="", lines=None, source_lang="fr", target_lang="en",
                  text_class="text_bubble", region_metadata=normalize_region_metadata({}))
    values.update(changes)
    return SimpleNamespace(**values)


def rendered(**changes):
    values = dict(text="<p>Hello</p>", font_family="Arial", font_size=12,
                  position=(10, 20), width=90, height=70, bold=False,
                  layout_preferences={"mode": "box", "min_font_size": 10})
    values.update(changes)
    return values


def kinds(issues):
    return {(issue.kind, issue.stage) for issue in issues}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class ReviewEvaluationTests(unittest.TestCase):
    def test_uncertain_ocr_tracks_this_reading_and_its_evidence(self):
        block = region()
        def set_scores(*scores):
            block.ocr_evidence = make_ocr_evidence(block.text, 'PP-OCR', 'ppocr-v5-rec-latin-mobile',
                                                   line_scores=scores)
        def flags():
            return [issue for issue in evaluate_region('page', block) if issue.kind == 'uncertain_ocr']
        set_scores(0.98, 0.61)
        issue, = flags()
        self.assertEqual(issue.stage, 'source')
        block.region_metadata = dismiss_issue(block.region_metadata, issue)
        self.assertEqual(flags(), [])
        set_scores(0.7)
        self.assertNotEqual(flags()[0].fingerprint, issue.fingerprint)
        block.region_metadata = mark_reviewed(block.region_metadata, 'source', fingerprint_for_stage('source', block))
        self.assertEqual(flags(), [])
        block.region_metadata = clear_review(block.region_metadata, 'source')
        block.text = 'Manually corrected'
        self.assertEqual(flags(), [])
        for score in (0.75, 0.99):
            set_scores(score)
            self.assertEqual(flags(), [])
        block.ocr_evidence = None
        self.assertEqual(flags(), [])

    def test_undetected_page_is_not_invented_as_missing_ocr(self):
        self.assertEqual(evaluate_region("page", None), [])

    def test_empty_region_only_has_missing_source(self):
        block = region(text=" \n", translation="")
        self.assertEqual(kinds(evaluate_region("page", block)), {("missing_ocr", "source")})

    def test_stage_requirements_do_not_flag_unperformed_downstream_work(self):
        source = region(translation="")
        self.assertEqual(kinds(evaluate_region("page", source)), {
            ("awaiting_review", "source"), ("missing_translation", "translation")})
        target_only = region(text="")
        self.assertEqual(kinds(evaluate_region("page", target_only)), {
            ("missing_ocr", "source"), ("awaiting_review", "translation")})
        self.assertEqual(kinds(evaluate_region("page", region())), {
            ("awaiting_review", "source"), ("awaiting_review", "translation")})

    def test_unknown_measurements_are_not_fit_success(self):
        for diagnostics in (None, {}, {"fits": None}, {"fits": False, "overflow_reasons": ["unsupported_content"]}):
            with self.subTest(diagnostics=diagnostics):
                issues = evaluate_region("page", region(), rendered(), diagnostics)
                layout = [item for item in issues if item.stage == "layout"]
                self.assertEqual(len(layout), 1)
                self.assertEqual(layout[0].kind, "awaiting_review")
                self.assertEqual(layout[0].detail, "diagnostics_unknown")

    def test_measured_diagnostics_support_mapping_or_result_object(self):
        values = {"fits": False, "overflow_reasons": ("width", "boundary"), "below_minimum": True,
                  "minimum_font_size": 14, "font_size": 12}
        mapping_result = evaluate_region("page", region(), rendered(), values)
        object_result = evaluate_region("page", region(), SimpleNamespace(**rendered()), SimpleNamespace(**values))
        self.assertEqual(mapping_result, object_result)
        self.assertIn(("overflow", "layout"), kinds(mapping_result))
        self.assertIn(("small_text", "layout"), kinds(mapping_result))
        for item in mapping_result:
            self.assertEqual(len(item.fingerprint), 64)

    def test_explicit_no_fit_is_reported_without_fabricated_geometry(self):
        issues = evaluate_region("page", region(), rendered(), {"fits": False})
        overflow = next(item for item in issues if item.kind == "overflow")
        self.assertEqual(overflow.detail, "does_not_fit")

    def test_empty_rich_document_has_no_layout_review_or_measurement_issue(self):
        text = '<html><head><style>p {color:black;}</style></head><body><p>&nbsp;<br /></p></body></html>'
        issues = evaluate_region("page", region(), rendered(text=text), {"fits": False, "below_minimum": True})
        self.assertFalse(any(item.stage == "layout" for item in issues))
        self.assertTrue(any(item.stage == "layout" for item in evaluate_region("page", region(), rendered(text="日本語"))))

    def test_free_text_and_protections_do_not_silently_approve_content(self):
        block = region(text_class="text_free", region_metadata={"protections": {
            "keep_source": True, "keep_translation": True, "keep_layout": True, "skip_auto_cleanup": True}})
        self.assertEqual(evaluate_region("page", block, rendered()), evaluate_region("page", region(), rendered()))

    def test_cleanup_requires_explicit_evidence_and_token_changes_invalidate_review(self):
        block = region()
        for workflow in (None, {"inpainted": True}, {"cleanup_done": True}, {"cleanup_fingerprint": " "}):
            self.assertFalse(any(issue.stage == "cleanup" for issue in evaluate_region("page", block, workflow=workflow)))
        workflow = {"cleanup_fingerprint": digest("patch-generation-1")}
        issue = next(issue for issue in evaluate_region("page", block, workflow=workflow) if issue.stage == "cleanup")
        block.region_metadata = mark_reviewed(block.region_metadata, "cleanup", issue.fingerprint)
        self.assertFalse(any(issue.stage == "cleanup" for issue in evaluate_region("page", block, workflow=workflow)))
        changed = {"cleanup_fingerprint": digest("patch-generation-2")}
        self.assertTrue(any(issue.stage == "cleanup" for issue in evaluate_region("page", block, workflow=changed)))

    def test_evaluation_does_not_mutate_inputs(self):
        block, props = region(), rendered()
        diagnostics = {"fits": False, "overflow_reasons": ["height"], "below_minimum": True}
        before = copy.deepcopy((block.__dict__, props, diagnostics))
        evaluate_region("page", block, props, diagnostics)
        self.assertEqual((block.__dict__, props, diagnostics), before)

    def test_issue_is_immutable_and_stable_key_excludes_content_and_path(self):
        block = region(translation="")
        original = next(item for item in evaluate_region("page", block, workflow={"page_path": "old.png"}) if item.kind == "missing_translation")
        block.text = "Salut"
        changed = next(item for item in evaluate_region("page", block, workflow={"page_path": "new.png"}) if item.kind == "missing_translation")
        self.assertEqual(original.key, changed.key)
        self.assertNotEqual(original.fingerprint, changed.fingerprint)
        self.assertEqual(changed.page_path, "new.png")
        with self.assertRaises(FrozenInstanceError):
            original.kind = "other"


class ReviewFingerprintTests(unittest.TestCase):
    def fingerprints(self, block, props=None, **kwargs):
        return {stage: fingerprint_for_stage(stage, block, props, **kwargs)
                for stage in ("source", "translation", "layout", "cleanup")}

    def test_source_translation_layout_changes_invalidate_only_relevant_stages(self):
        block, props = region(), rendered()
        before = self.fingerprints(block, props)
        block.text = "Autre source"
        source = self.fingerprints(block, props)
        self.assertNotEqual(before["source"], source["source"])
        self.assertNotEqual(before["translation"], source["translation"])
        self.assertEqual(before["layout"], source["layout"])
        block.translation = "New translation"
        target = self.fingerprints(block, props)
        self.assertEqual(source["source"], target["source"])
        self.assertNotEqual(source["translation"], target["translation"])
        self.assertNotEqual(source["layout"], target["layout"])
        props["text"] = "<p><b>New</b> translation</p>"
        formatted = self.fingerprints(block, props)
        self.assertEqual(target["source"], formatted["source"])
        self.assertEqual(target["translation"], formatted["translation"])
        self.assertNotEqual(target["layout"], formatted["layout"])

    def test_source_geometry_changes_source_and_fit_review_but_not_wording_review(self):
        block, props = region(), rendered()
        before = self.fingerprints(block, props)
        block.xyxy[0] += 10
        after = self.fingerprints(block, props)
        self.assertNotEqual(before["source"], after["source"])
        self.assertNotEqual(before["layout"], after["layout"])
        self.assertEqual(before["translation"], after["translation"])
        before = after
        block.bubble_xyxy[2] += 10
        after = self.fingerprints(block, props)
        self.assertNotEqual(before["layout"], after["layout"])
        self.assertEqual(before["source"], after["source"])

    def test_language_changes_have_stage_specific_scope(self):
        block = region()
        before = self.fingerprints(block)
        target = self.fingerprints(block, languages={"source": "fr", "target": "vi"})
        self.assertEqual(before["source"], target["source"])
        self.assertNotEqual(before["translation"], target["translation"])
        source = self.fingerprints(block, languages={"source_lang": "es", "target_lang": "en"})
        self.assertNotEqual(before["source"], source["source"])
        self.assertNotEqual(before["translation"], source["translation"])

    def test_ids_paths_reviews_and_unrelated_settings_do_not_change_fingerprints(self):
        block, props = region(), rendered()
        before = self.fingerprints(block, props)
        block.region_id = "new-copy-id"
        block.region_metadata = mark_reviewed({"protections": {"keep_layout": True}}, "source", before["source"])
        props["region_id"] = "different-link"
        props["irrelevant_gui_selection"] = True
        after = self.fingerprints(block, props, workflow={"page_path": "elsewhere.png"})
        self.assertEqual(before, after)

    def test_layout_style_position_and_preferences_each_invalidate_review(self):
        block, props = region(), rendered()
        before = fingerprint_for_stage("layout", block, props)
        for key, value in {"font_family": "Other Font", "font_size": 13, "position": (30, 40),
                           "width": 95, "height": 80, "document_font": "font-state", "italic": True,
                           "layout_preferences": {"mode": "bubble", "padding": 8}}.items():
            with self.subTest(key=key):
                changed = {**props, key: value}
                self.assertNotEqual(before, fingerprint_for_stage("layout", block, changed))

    def test_integer_float_and_tuple_list_roundtrips_are_equivalent(self):
        first, second = region(), region()
        second.xyxy = tuple(float(number) for number in first.xyxy)
        self.assertEqual(fingerprint_for_stage("source", first), fingerprint_for_stage("source", second))
        first.text, second.text = "Cafe\u0301\r\nBonjour", "Café\nBonjour"
        self.assertEqual(fingerprint_for_stage("source", first), fingerprint_for_stage("source", second))


class ReviewDecisionTests(unittest.TestCase):
    def test_intentionally_missing_source_or_translation_can_be_reviewed(self):
        for source, target, stage in (("", "", "source"), ("Bonjour", "", "translation")):
            block = region(text=source, translation=target)
            fingerprint = fingerprint_for_stage(stage, block)
            block.region_metadata = mark_reviewed(block.region_metadata, stage, fingerprint)
            self.assertFalse(any(issue.stage == stage for issue in evaluate_region("page", block)))
            block.region_metadata = clear_review(block.region_metadata, stage)
            self.assertTrue(any(issue.stage == stage for issue in evaluate_region("page", block)))

    def test_review_becomes_stale_after_translation_edit(self):
        block = region()
        block.region_metadata = mark_reviewed(block.region_metadata, "translation", fingerprint_for_stage("translation", block))
        self.assertNotIn(("awaiting_review", "translation"), kinds(evaluate_region("page", block)))
        block.translation = "Good day"
        self.assertIn(("awaiting_review", "translation"), kinds(evaluate_region("page", block)))

    def test_layout_approval_does_not_hide_measured_problems(self):
        block, props = region(), rendered()
        diagnostic = {"fits": False, "overflow_reasons": ["height"], "below_minimum": True}
        block.region_metadata = mark_reviewed(block.region_metadata, "layout", fingerprint_for_stage("layout", block, props))
        issues = evaluate_region("page", block, props, diagnostic)
        self.assertNotIn(("awaiting_review", "layout"), kinds(issues))
        self.assertIn(("overflow", "layout"), kinds(issues))
        self.assertIn(("small_text", "layout"), kinds(issues))

    def test_dismissal_is_specific_and_invalidated_by_threshold_or_layout_change(self):
        block, props = region(), rendered()
        diagnostics = {"fits": False, "overflow_reasons": ["width"], "below_minimum": True,
                       "minimum_font_size": 14}
        issues = evaluate_region("page", block, props, diagnostics)
        small = next(issue for issue in issues if issue.kind == "small_text")
        block.region_metadata = dismiss_issue(block.region_metadata, small)
        self.assertNotIn(("small_text", "layout"), kinds(evaluate_region("page", block, props, diagnostics)))
        self.assertIn(("overflow", "layout"), kinds(evaluate_region("page", block, props, diagnostics)))
        diagnostics["minimum_font_size"] = 15
        self.assertIn(("small_text", "layout"), kinds(evaluate_region("page", block, props, diagnostics)))
        diagnostics["minimum_font_size"] = 14
        props["font_size"] = 11
        self.assertIn(("small_text", "layout"), kinds(evaluate_region("page", block, props, diagnostics)))

    def test_awaiting_dismissals_for_multiple_stages_coexist(self):
        block, props = region(), rendered()
        for issue in evaluate_region("page", block, props):
            block.region_metadata = dismiss_issue(block.region_metadata, issue)
        self.assertEqual(evaluate_region("page", block, props), [])
        block.translation = "Updated words"
        issues = evaluate_region("page", block, props)
        self.assertNotIn(("awaiting_review", "source"), kinds(issues))
        self.assertIn(("awaiting_review", "translation"), kinds(issues))
        self.assertIn(("awaiting_review", "layout"), kinds(issues))

    def test_helpers_return_independent_metadata_without_changing_protection(self):
        metadata = normalize_region_metadata({"protections": {"keep_layout": True}, "layout_preferences": {"mode": "bubble"}})
        approved = mark_reviewed(metadata, "layout", digest("layout"))
        self.assertEqual(metadata["review"]["approved_fingerprints"], {})
        self.assertTrue(approved["protections"]["keep_layout"])
        approved["layout_preferences"]["mode"] = "box"
        self.assertEqual(metadata["layout_preferences"]["mode"], "bubble")
        for stage, value in (("unknown", digest("valid")), ("source", "not a hash")):
            with self.assertRaises(ValueError):
                mark_reviewed(metadata, stage, value)


class ReviewMetadataTests(unittest.TestCase):
    def test_old_defaults_and_invalid_data_are_safe(self):
        for value in (None, {}, "bad", {"review": {"approved_fingerprints": [], "dismissed_issue_fingerprints": 5}}):
            result = normalize_region_metadata(value)
            self.assertEqual(result["review"], {"approved_fingerprints": {}, "dismissed_issue_fingerprints": {}})
            self.assertEqual(result["schema_version"], 1)
            self.assertFalse(any(result["protections"].values()))

    def test_only_known_keys_and_bounded_hashes_survive(self):
        hashes = [digest(str(index)) for index in range(20)]
        value = {"approved_fingerprints": {"source": hashes[0].upper(), "layout": "not a hash", "invented": hashes[1]},
                 "dismissed_issue_fingerprints": {"overflow": hashes, "awaiting_review": hashes[1].upper(),
                                                  "small_text": ["a" * 100000, 5], "invented": hashes}}
        normalized = normalize_review_metadata(value)
        self.assertEqual(normalized["approved_fingerprints"], {"source": hashes[0]})
        self.assertEqual(normalized["dismissed_issue_fingerprints"], {
            "overflow": hashes[-MAX_DISMISSED_FINGERPRINTS:], "awaiting_review": [hashes[1]]})

    def test_snapshot_and_project_codec_preserve_reviews_but_duplicate_resets_them(self):
        import msgpack
        from app.projects.parsers import ProjectEncoder, ProjectDecoder
        from modules.utils.textblock import TextBlock
        original = TextBlock(text_bbox=[10, 20, 100, 90], text="Bonjour", translation="Hello")
        original.region_metadata = mark_reviewed({"protections": {"keep_translation": True},
                                                  "layout_preferences": {"mode": "bubble"}},
                                                 "translation", fingerprint_for_stage("translation", original))
        encoded = msgpack.packb(original, default=ProjectEncoder().encode, use_bin_type=True)
        decoded = msgpack.unpackb(encoded, object_hook=ProjectDecoder().decode)
        for snapshot in (original.deep_copy(), decoded):
            self.assertEqual(snapshot.region_metadata, original.region_metadata)
            self.assertEqual(snapshot.region_id, original.region_id)
        duplicate = original.duplicate()
        self.assertNotEqual(duplicate.region_id, original.region_id)
        self.assertEqual(duplicate.region_metadata["review"], normalize_review_metadata(None))
        self.assertFalse(duplicate.region_metadata["protections"]["keep_translation"])
        self.assertEqual(duplicate.region_metadata["layout_preferences"], {"mode": "bubble"})

    def test_evaluation_creates_no_qt_application_or_model_runtime(self):
        # Existing modules.utils.__init__ imports TextBlock and therefore QtCore
        # and numpy. The pure evaluator must still need no GUI/application or
        # model runtime, and evaluating a plain value object must create none.
        code = """
import sys
from types import SimpleNamespace
from modules.review.issues import evaluate_region
from PySide6.QtCore import QCoreApplication
assert QCoreApplication.instance() is None
assert evaluate_region('page', SimpleNamespace(region_id='r', text='', translation=''))
assert QCoreApplication.instance() is None
assert not any(k.startswith(('PySide6.QtWidgets', 'torch', 'diffusers', 'transformers')) for k in sys.modules)
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
