"""Delivered-image geometry, immutable request context and opt-in behavior."""
import copy
import json
import unittest
from types import SimpleNamespace

import numpy as np

from modules.translation.request import build_request, StaleTranslationRequest, TranslationContextError
from modules.translation.spatial import (SpatialContextError, build_visual_reference,
                                         geometry_signature, region_polygon)
from modules.utils.textblock import TextBlock


def engine(**changes):
    values = dict(source_lang='English', target_lang='French', model='fixture',
                  spatial_context_enabled=True, img_as_llm_input=True,
                  supports_images=True, max_tokens=512,
                  get_system_prompt=lambda *_: 'Translate the supplied OCR.')
    return SimpleNamespace(**(values | changes))


def block(rect=(10, 20, 70, 80), **kwargs):
    return TextBlock(text_bbox=np.array(rect, dtype=float), text='Hello', **kwargs)


class SpatialContextTests(unittest.TestCase):
    def test_normalized_bounds_use_delivered_image_dimensions(self):
        reference = build_visual_reference([block()], (100, 200, 3), ['block_0'])
        self.assertEqual((reference['image_width'], reference['image_height']), (200, 100))
        self.assertEqual(reference['regions'][0], dict(id='block_0', bbox_normalized=[.05, .2, .35, .8],
                                                       angle=0., reading_order=0, clipped=False, visible=True))

    def test_tiny_box_keeps_positive_area_and_outward_edges_at_prompt_precision(self):
        # Both horizontal edges would collapse to one coordinate under nearest rounding.
        rect = (123.451, 75.001, 123.452, 75.002)
        reference = build_visual_reference([block(rect)], (1000, 1000), ['tiny'])
        row = reference['regions'][0]
        self.assertTrue(row['visible'])
        self.assertFalse(row['clipped'])
        self.assertEqual(row['bbox_normalized'], [.12345, .075, .12346, .07501])

    def test_large_axis_precision_contains_rotated_polygon_with_subpixel_error(self):
        height, width = 1_234_567, 1024
        item = block((120.52, 483247.325, 150.91, 483275.536), angle=23)
        points = region_polygon(item)
        xs, ys = zip(*points)
        raw = [min(xs)/width, min(ys)/height, max(xs)/width, max(ys)/height]
        wire = build_visual_reference([item], (height, width), ['tall'])['regions'][0]['bbox_normalized']
        # Strictly enclose the actual rotated rectangle; each edge can grow at most 0.1 px.
        for i, (edge, exact, length) in enumerate(zip(wire, raw, (width, height, width, height))):
            outward = (exact-edge) if i < 2 else (edge-exact)
            self.assertGreaterEqual(outward, 0)
            self.assertLessEqual(outward*length, 0.1)
        self.assertEqual(wire[0], round(wire[0], 5))
        self.assertEqual(wire[1], round(wire[1], 8))

    def test_position_labels_cover_all_nine_thirds_and_use_unrounded_midpoint(self):
        centers = (50, 150, 250)
        boxes = [block((x-3, y-3, x+3, y+3)) for y in centers for x in centers]
        # This center lies just left of the first divider even though its rounded bounds straddle it.
        boxes.append(block((97.12345, 15, 102.87654, 25)))
        rows = build_visual_reference(boxes, (300, 300), [str(i) for i in range(10)])['regions']
        self.assertEqual([r['position'] for r in rows[:9]],
                         ['top left', 'top center', 'top right', 'middle left', 'center', 'middle right',
                          'bottom left', 'bottom center', 'bottom right'])
        compact = rows[9]['bbox_normalized']
        self.assertGreater((compact[0]+compact[2])/2, 1/3)
        self.assertEqual(rows[9]['position'], 'top left')

    def test_position_uses_clipped_visible_extent_and_survives_uniform_scaling(self):
        rect = (-60, 25, 90, 50)
        for scale in (1, 4):
            with self.subTest(scale=scale):
                boxes = [block(np.asarray(rect)*scale), block(np.array([125, 25, 170, 50])*scale)]
                visible, hidden = build_visual_reference(boxes, (90*scale, 120*scale), ['a','b'])['regions']
                self.assertTrue(visible['clipped'])
                self.assertTrue(visible['visible'])
                self.assertEqual(visible['position'], 'center')
                self.assertTrue(hidden['clipped'])
                self.assertFalse(hidden['visible'])
                self.assertIsNone(hidden['bbox_normalized'])
                self.assertNotIn('position', hidden)

    def test_only_visible_repeated_nonempty_ocr_gets_a_position_hint(self):
        texts = ['  Cafe\u0301\n au lait!', 'ＣＡＦÉ AU\tLAIT!', 'café au lait?', 'unique', '   ', '\t', 'café au lait!']
        boxes = [block((10+i*35, 10, 30+i*35, 22)) for i in range(len(texts))]
        for item, text in zip(boxes, texts):
            item.text = text
        boxes[-1].xyxy = (310, 12, 333, 25)  # Repeated, but outside the 300 px wide image.
        request = build_request(engine(), boxes, np.zeros((90, 300, 3), np.uint8))
        rows = json.loads(request.spatial_reference_json)['regions']
        self.assertEqual([r.get('position') for r in rows], ['top left', 'top left', None, None, None, None, None])
        self.assertFalse(rows[-1]['visible'])
        self.assertEqual(request.source_texts, tuple(texts))
        self.assertEqual(list(json.loads(request.user_prompt.split('Translate this JSON:\n')[-1]).values()), texts)

    def test_rotated_box_uses_item_local_pivot_and_clockwise_angles(self):
        item = block((10, 20, 50, 40), angle=90)
        np.testing.assert_allclose(region_polygon(item), ((40, 10), (40, 50), (20, 50), (20, 10)))
        item.tr_origin_point = (0, 0)
        np.testing.assert_allclose(region_polygon(item), ((10, 20), (10, 60), (-10, 60), (-10, 20)), atol=1e-10)
        row = build_visual_reference([item], (100, 100), ['b'])['regions'][0]
        np.testing.assert_allclose(row['bbox_normalized'], (0, .2, .1, .6), atol=1e-10)
        self.assertTrue(row['clipped'])
        self.assertEqual(row['angle'], 90)

    def test_partial_border_and_fully_outside_keep_every_id(self):
        rows = build_visual_reference([block((-10, 20, 30, 110)), block((150, 20, 180, 50)),
                                       block((100, 0, 110, 20))], (100, 100), ['a', 'b', 'c'])['regions']
        self.assertEqual([row['id'] for row in rows], ['a', 'b', 'c'])
        self.assertEqual(rows[0]['bbox_normalized'], [0., .2, .3, 1.])
        self.assertTrue(all(row['clipped'] for row in rows))
        self.assertEqual([row['visible'] for row in rows], [True, False, False])
        self.assertIsNone(rows[1]['bbox_normalized'])
        self.assertIsNone(rows[2]['bbox_normalized'])

    def test_rotation_does_not_report_false_intersection_from_envelope(self):
        # This diamond's envelope reaches the canvas corner, but its ink region
        # does not; clipping only an axis-aligned envelope would be misleading.
        item = block((-30, -30, -2, -2), angle=45)
        row = build_visual_reference([item], (100, 100), ['a'])['regions'][0]
        self.assertFalse(row['visible'])
        self.assertIsNone(row['bbox_normalized'])

    def test_uniform_resize_and_crop_offsets_match_delivered_pixels(self):
        original = block((20, 40, 100, 80), angle=25)
        original.tr_origin_point = (10, 15)
        resized = original.deep_copy()
        resized.xyxy *= 2
        resized.tr_origin_point = (20, 30)
        a = build_visual_reference([original], (200, 300), ['a'])['regions'][0]
        b = build_visual_reference([resized], (400, 600), ['a'])['regions'][0]
        np.testing.assert_allclose(a['bbox_normalized'], b['bbox_normalized'])
        cropped = block((-10, 10, 70, 50))  # source box minus crop origin (30,30)
        row = build_visual_reference([cropped], (60, 100), ['a'])['regions'][0]
        self.assertEqual(row['bbox_normalized'], [0., .16666, .7, .83334])
        self.assertTrue(row['clipped'])

    def test_invalid_geometry_is_rejected_without_modifying_regions(self):
        for rect in (None, (0, 0, 0, 20), (20, 0, 10, 20), (0, 0, float('nan'), 20),
                     (0, 0, float('inf'), 20), (0, 1, 2), ('0', 0, 10, 20)):
            with self.subTest(rect=rect):
                item = TextBlock(text='Hello', text_bbox=rect)
                with self.assertRaises(SpatialContextError):
                    build_visual_reference([item], (100, 100), ['a'])
                self.assertEqual(item.text, 'Hello')
        item = block()
        for field, value in (('angle', float('nan')), ('tr_origin_point', (0, float('inf')))):
            setattr(item, field, value)
            with self.assertRaises(SpatialContextError):
                geometry_signature(item)
            item = block()

    def test_inactive_paths_do_not_validate_geometry_or_send_coordinates(self):
        item = TextBlock(text='Hello')  # deliberately no geometry
        pixels = np.zeros((100, 200, 3), np.uint8)
        for config, image, supported in ((engine(spatial_context_enabled=False), pixels, True),
                                         (engine(img_as_llm_input=False), pixels, True),
                                         (engine(supports_images=False), pixels, True),
                                         (engine(), None, True), (engine(), pixels, False)):
            request = build_request(config, [item], image, structured_context_supported=supported)
            self.assertEqual(request.spatial_reference_json, '')
            self.assertEqual(request.spatial_geometry, ())
            self.assertNotIn('Visual reference data:', request.user_prompt)

    def test_effective_geometry_and_enablement_affect_cache_identity(self):
        item, config, pixels = block(), engine(), np.zeros((100, 200, 3), np.uint8)
        before = build_request(config, [item], pixels)
        item.xyxy[0] += .0002  # This subpixel change stays inside the same compact prompt cell.
        changed = build_request(config, [item], pixels)
        self.assertEqual(before.spatial_reference_json, changed.spatial_reference_json)
        self.assertEqual(before.image_bytes, changed.image_bytes)
        self.assertNotEqual(before.spatial_geometry, changed.spatial_geometry)
        self.assertNotEqual(before.fingerprint, changed.fingerprint)
        with self.assertRaises(StaleTranslationRequest):
            before.assert_current([item])
        config.spatial_context_enabled = False
        disabled = build_request(config, [item], pixels)
        self.assertNotEqual(before.fingerprint, disabled.fingerprint)
        item.xyxy[0] += 5
        ordinary = build_request(config, [item], pixels)
        self.assertEqual(disabled.fingerprint, ordinary.fingerprint)
        self.assertEqual(disabled.user_prompt, ordinary.user_prompt)
        self.assertEqual(ordinary.image_bytes, pixels.tobytes())
        disabled.assert_current([item])
        config.img_as_llm_input = False
        text_only = build_request(config, [item], pixels)
        config.spatial_context_enabled = True
        self.assertEqual(text_only.fingerprint, build_request(config, [item], pixels).fingerprint)

    def test_snapshot_and_response_keys_remain_stable_after_reordering(self):
        first, second = block(), block((100, 20, 150, 80))
        pixels = np.zeros((100, 200, 3), np.uint8)
        request = build_request(engine(), [first, second], pixels)
        payload = json.loads(request.spatial_reference_json)
        self.assertEqual([r['id'] for r in payload['regions']], ['block_0', 'block_1'])
        self.assertEqual([r['reading_order'] for r in payload['regions']], [0, 1])
        self.assertEqual(json.loads(request.user_prompt.split('Translate this JSON:\n')[-1]),
                         {'block_0': 'Hello', 'block_1': 'Hello'})
        self.assertEqual(set(request.assert_current([second, first])), {first.region_id, second.region_id})
        payload['regions'][0]['bbox_normalized'][0] = .9
        self.assertEqual(json.loads(request.spatial_reference_json)['regions'][0]['bbox_normalized'][0], .05)
        pixels[:] = 100
        self.assertTrue(np.all(request.image == 0))
        self.assertIn('Text-box coordinates: included (experimental)', request.preview)

    def test_geometry_changes_reject_stale_results_only_for_spatial_requests(self):
        for field, changed in (('xyxy', (15, 20, 70, 80)), ('angle', 15),
                               ('tr_origin_point', (0, 0))):
            with self.subTest(field=field):
                item = block()
                before = build_request(engine(), [item], np.zeros((100, 200, 3), np.uint8))
                ordinary = build_request(engine(spatial_context_enabled=False), [item])
                setattr(item, field, changed)
                with self.assertRaises(StaleTranslationRequest):
                    before.assert_current([item])
                ordinary.assert_current([item])

    def test_coordinate_tokens_are_included_in_context_budget(self):
        items = [block() for _ in range(12)]
        pixels = np.zeros((100, 200, 3), np.uint8)
        config = engine(spatial_context_enabled=False)
        ordinary = build_request(config, items, pixels)
        config.context_size = ordinary.prompt_estimated_tokens + ordinary.output_tokens_reserved + 256
        build_request(config, items, pixels)
        config.spatial_context_enabled = True
        with self.assertRaises(TranslationContextError):
            build_request(config, items, pixels)


if __name__ == '__main__':
    unittest.main()
