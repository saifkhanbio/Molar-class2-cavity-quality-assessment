"""Run with python3 -m unittest discover -s tests -v."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from cus_ist_ratio import (analyze_record, closest_ratio, distance_rows, fallback_ratio,
                           save_overlay, summary_row, validate_isthmus)
from isthmus_disc import Config, detect_isthmuses
from scipy import ndimage as ndi
from cusp_geometry import ALTERNATE_CUSP_PAIRS, segments_intersect


def fixtures():
    necks = [
        {"isthmus_id": 1, "center_rc": [70, 50], "endpoints_rc": [[70, 45], [70, 55]],
         "width_px": 10, "component_id": 1, "is_fallback": False, "detection_method": "geometric_neck"},
        {"isthmus_id": 2, "center_rc": [148, 50], "endpoints_rc": [[148, 40], [148, 60]],
         "width_px": 20, "component_id": 1, "is_fallback": False, "detection_method": "geometric_neck"},
    ]
    pairs = [{"pair_id": 11, "region_ids": [1, 3], "centers_rc": [[50, 10], [50, 90]], "opposite_sides_valid": True},
             {"pair_id": 12, "region_ids": [2, 4], "centers_rc": [[150, 20], [150, 80]], "opposite_sides_valid": True}]
    sites = [{"pair_id": 11, "endpoints_rc": [[50, 35], [50, 65]], "component_id": 1, "status": "available"},
             {"pair_id": 12, "endpoints_rc": [[150, 35], [150, 65]], "component_id": 1, "status": "available"}]
    return necks, pairs, sites


class RatioTests(unittest.TestCase):
    def test_crossing_alternatives_are_used_by_both_matching_stages(self):
        necks, original, original_sites = fixtures()
        alternatives = copy.deepcopy(original)
        alternatives[0].update(pair_id=21, region_ids=[1, 4])
        alternatives[1].update(pair_id=22, region_ids=[2, 3])
        sites = [dict(site, pair_id=pair['pair_id']) for site, pair in zip(original_sites, alternatives)]
        selected, candidates = closest_ratio(necks, original + alternatives, original_sites + sites,
                                              Config(), active_pairs=ALTERNATE_CUSP_PAIRS)
        self.assertEqual(selected['cusp_pair_label'], 'C2-C3')
        self.assertEqual({c['cusp_pair_label'] for c in candidates}, {'C1-C4', 'C2-C3'})
        selected, candidates = fallback_ratio(necks, original + alternatives, [], Config(), [],
                                               active_pairs=ALTERNATE_CUSP_PAIRS)
        self.assertEqual(selected['cusp_pair_label'], 'C2-C3')
        self.assertEqual({c['cusp_pair_label'] for c in candidates}, {'C1-C4', 'C2-C3'})
    def test_irrelevant_pair_is_excluded_even_if_closest(self):
        necks, pairs, sites = fixtures()
        pairs.append({"pair_id": 99, "region_ids": [1, 2], "centers_rc": [[70, 10], [70, 90]]})
        sites.append({"pair_id": 99, "endpoints_rc": [[70, 45], [70, 55]],
                      "component_id": 1, "status": "available"})
        selected, candidates = closest_ratio(necks[:1], pairs, sites, Config())
        self.assertEqual(selected["cusp_pair_label"], "C1-C3")
        self.assertNotIn(99, [row['cusp_pair_id'] for row in candidates])

    def test_no_allowed_valid_crossing_never_substitutes_another_pair(self):
        necks, pairs, sites = fixtures()
        pairs[0]['region_ids'] = [1, 4]
        pairs[1]['region_ids'] = [2, 3]
        selected, candidates = closest_ratio(necks, pairs, sites, Config())
        self.assertIsNone(selected)
        self.assertEqual(candidates, [])

    def test_two_by_two_chooses_closest_not_smallest_ratio(self):
        necks, pairs, sites = fixtures()
        chosen, candidates = closest_ratio(necks, pairs, sites, Config())
        self.assertEqual(len(candidates), 4)
        self.assertEqual(sum(item["selected"] for item in candidates), 1)
        self.assertEqual((chosen["isthmus_id"], chosen["cusp_pair_id"]), (2, 12))
        self.assertAlmostEqual(chosen["ratio_isthmus_to_intercuspal"], 20 / 60)
        self.assertAlmostEqual(chosen["ratio_intercuspal_to_isthmus"], 3)
        self.assertGreater(chosen["ratio_isthmus_to_intercuspal"], min(c["ratio_isthmus_to_intercuspal"] for c in candidates))
        shuffled, _ = closest_ratio(necks[::-1], pairs[::-1], sites[::-1], Config())
        self.assertEqual(chosen, shuffled)

    def test_single_width_matches_nearest_pair(self):
        necks, pairs, sites = fixtures()
        chosen, _ = closest_ratio(necks[:1], pairs, sites, Config())
        self.assertEqual(chosen["cusp_pair_id"], 11)
        self.assertAlmostEqual(chosen["ratio_isthmus_to_intercuspal"], 0.125)
        self.assertFalse(chosen["within_near_tolerance"])

    def test_one_distance_selects_closest_of_two_isthmuses(self):
        necks, pairs, sites = fixtures()
        chosen, _ = closest_ratio(necks, pairs[1:], sites[1:], Config())
        self.assertEqual(chosen["isthmus_id"], 2)

    def test_near_only_withholds_far_match(self):
        necks, pairs, sites = fixtures()
        chosen, candidates = closest_ratio(necks[:1], pairs, sites, Config(), "near-only")
        self.assertIsNone(chosen)
        self.assertTrue(all(not item["eligible"] for item in candidates))

    def test_invalid_crossings_and_other_components_are_not_candidates(self):
        necks, pairs, sites = fixtures()
        sites[0]["status"] = "no_cavity_between_cusps"
        sites[1]["component_id"] = 2
        chosen, candidates = closest_ratio(necks, pairs, sites, Config())
        self.assertIsNone(chosen)
        self.assertEqual(candidates, [])

    def test_zero_length_cusp_pair_does_not_divide_by_zero(self):
        necks, pairs, sites = fixtures()
        pairs[0]["centers_rc"] = [[50, 10], [50, 10]]
        chosen, candidates = closest_ratio(necks, pairs[:1], sites[:1], Config())
        self.assertIsNone(chosen)
        self.assertEqual(candidates, [])

    def test_common_scale_and_rotation_preserve_ratio(self):
        necks, pairs, sites = fixtures()
        reference, _ = closest_ratio(necks, pairs, sites, Config())
        for scale in (1, 2):
            n, p, s = copy.deepcopy((necks, pairs, sites))
            for item in n + p + s:
                for key in ("endpoints_rc", "centers_rc", "center_rc"):
                    if key in item:
                        # Rotate by 90 degrees, scale uniformly, and translate.
                        coords = np.asarray(item[key])
                        item[key] = (coords[..., ::-1] * [scale, -scale] + [0, 400]).tolist()
                if "width_px" in item:
                    item["width_px"] *= scale
            selected, _ = closest_ratio(n, p, s, Config())
            self.assertEqual(selected["cusp_pair_id"], reference["cusp_pair_id"])
            self.assertAlmostEqual(selected["ratio_isthmus_to_intercuspal"], reference["ratio_isthmus_to_intercuspal"])

    def test_calibration_changes_lengths_not_dimensionless_ratio(self):
        necks, pairs, sites = fixtures()
        plain, _ = closest_ratio(necks, pairs, sites, Config())
        calibrated, _ = closest_ratio(necks, pairs, sites, Config(pixel_size_mm=0.05))
        self.assertIsNone(plain["intercuspal_distance_mm"])
        self.assertEqual(plain["ratio_isthmus_to_intercuspal"], calibrated["ratio_isthmus_to_intercuspal"])
        self.assertAlmostEqual(calibrated["isthmus_width_mm"], 1)
        self.assertAlmostEqual(calibrated["intercuspal_distance_mm"], 3)

    def test_bad_saved_width_is_rejected(self):
        neck = fixtures()[0][0]
        for width in (0, -1, float("nan"), 12):
            with self.subTest(width=width), self.assertRaises(ValueError):
                validate_isthmus({**neck, "width_px": width}, (200, 100))

    def test_missing_ratio_exports_blanks(self):
        row = summary_row({"filename": "sample.png", "status": "no_opposing_cusp_pair", "selected": None})
        self.assertIsNone(row["ratio_isthmus_to_intercuspal"])
        self.assertIsNone(row["intercuspal_distance_px"])


class RelaxedRatioTests(unittest.TestCase):
    def test_fallback_never_reintroduces_rejected_same_side_pairs(self):
        necks, pairs, sites = fixtures()
        for pair in pairs:
            pair['opposite_sides_valid'] = False
        self.assertEqual(closest_ratio(necks, pairs, sites, Config()), (None, []))
        self.assertEqual(fallback_ratio(necks, pairs, sites, Config(), []), (None, []))

    def test_legacy_candidates_without_side_validation_cannot_bypass_fallback(self):
        necks, pairs, sites = fixtures()
        _, strict = closest_ratio(necks, pairs, sites, Config(), 'near-only')
        for candidate in strict:
            candidate.pop('opposite_sides_valid')
        self.assertEqual(fallback_ratio(necks, pairs, sites, Config(), strict), (None, []))

    def test_missing_crossings_choose_closest_allowed_pair_across_both_widths(self):
        necks, pairs, sites = fixtures()
        for site in sites:
            site['status'] = 'no_cavity_between_cusps'
        pairs.append({'pair_id': 99, 'region_ids': [1, 2], 'centers_rc': [[148, 10], [148, 90]]})
        selected, candidates = fallback_ratio(necks, pairs, sites, Config(), [])
        self.assertEqual((selected['isthmus_id'], selected['cusp_pair_id']), (2, 12))
        self.assertEqual(len(candidates), 4)
        self.assertTrue(selected['ratio_is_fallback'])
        self.assertTrue(selected['crossing_checks_relaxed'])
        self.assertIsNone(selected['cavity_crossing_endpoints_rc'])
        self.assertEqual(selected['original_crossing_status'], 'no_cavity_between_cusps')
        self.assertEqual(selected['match_distance_basis'], 'finite_cusp_center_segment')
        self.assertAlmostEqual(selected['ratio_isthmus_to_intercuspal'], 20 / 60)

    def test_distant_pair_expands_tolerance_without_changing_measurements(self):
        necks, pairs, sites = fixtures()
        selected, _ = fallback_ratio(necks[:1], pairs[1:], sites, Config(), [])
        self.assertGreater(selected['proximity_relaxation_factor'], 1)
        self.assertFalse(selected['within_near_tolerance'])
        self.assertGreaterEqual(selected['relaxed_near_threshold_px'], selected['match_distance_px'])
        self.assertLess(selected['relaxed_near_threshold_px'] / 2, selected['match_distance_px'])
        self.assertEqual(selected['isthmus_width_px'], 10)
        self.assertEqual(selected['intercuspal_distance_px'], 60)

    def test_near_only_rejection_relaxes_proximity_before_crossing_checks(self):
        necks, pairs, sites = fixtures()
        selected, strict = closest_ratio(necks[:1], pairs[:1], sites[:1], Config(), 'near-only')
        self.assertIsNone(selected)
        selected, _ = fallback_ratio(necks[:1], pairs[:1], sites[:1], Config(), strict)
        self.assertEqual(selected['ratio_fallback_stage'], 'proximity_relaxed')
        self.assertFalse(selected['crossing_checks_relaxed'])
        self.assertEqual(selected['cavity_crossing_endpoints_rc'], sites[0]['endpoints_rc'])
        self.assertFalse(strict[0]['eligible'])

    def test_different_component_is_explicitly_flagged(self):
        necks, pairs, sites = fixtures()
        sites[0]['component_id'] = 9
        selected, _ = fallback_ratio(necks[:1], pairs[:1], sites[:1], Config(), [])
        self.assertEqual(selected['original_crossing_status'], 'crossing_in_different_component')
        self.assertTrue(selected['crossing_checks_relaxed'])

    def test_missing_or_invalid_measurements_are_not_invented(self):
        necks, pairs, sites = fixtures()
        for n, p in (([], pairs), (necks, [])):
            self.assertEqual(fallback_ratio(n, p, sites, Config(), []), (None, []))
        pairs[0]['centers_rc'] = [[10, 10], [10, 10]]
        self.assertEqual(fallback_ratio(necks, pairs[:1], sites, Config(), []), (None, []))


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        rows, cols = np.mgrid[:240, :180]
        self.mask = (rows >= 20) & (rows <= 219) & (np.abs(cols - 90) < 18)
        self.cusps = np.zeros_like(self.mask)
        for row, col in ((80, 45), (80, 135), (155, 135), (165, 45)):
            self.cusps |= ((rows - row) / 22) ** 2 + ((cols - col) / 12) ** 2 <= 1
        self.record = detect_isthmuses(self.mask, cusp_mask=self.cusps)
        self.record.update(filename="sample_mask.png", mask_shape=list(self.mask.shape))

    def test_new_fallback_width_is_used_and_flagged(self):
        result = analyze_record(self.record, self.mask, self.cusps, Config())
        chosen = result["selected"]
        self.assertIsNotNone(chosen)
        self.assertTrue(chosen["is_fallback"])
        self.assertEqual(chosen["isthmus_width_px"], self.record["isthmuses"][0]["width_px"])
        self.assertAlmostEqual(chosen["ratio_isthmus_to_intercuspal"], 35 / chosen["intercuspal_distance_px"], delta=0.01)
        self.assertEqual(result["status"], "ratio_requires_review")
        self.assertFalse(result["pair_recovery_used"])

    def test_dimension_mismatch_is_not_silently_resized(self):
        with self.assertRaises(ValueError):
            analyze_record(self.record, self.mask, self.cusps[:200], Config())

    def test_row_order_is_corrected_into_two_opposite_bank_distances(self):
        rows, cols = np.mgrid[:240, :180]
        cusps = np.zeros_like(self.mask)
        for row in (80, 160):
            for col in (45, 135):
                cusps |= ((rows - row) / 22) ** 2 + ((cols - col) / 12) ** 2 <= 1
        result = analyze_record(self.record, self.mask, cusps, Config(), enable_ratio_fallback=False)
        self.assertIsNotNone(result['selected'])
        self.assertTrue(result['selected']['opposite_sides_valid'])
        self.assertFalse(result['pair_recovery_used'])
        distances = distance_rows(result)
        self.assertEqual([r['cusp_pair_label'] for r in distances], ['C1-C3', 'C2-C4'])
        self.assertTrue(all(r['intercuspal_distance_px'] > 0 for r in distances))
        self.assertTrue(all(r['opposite_sides_valid'] for r in distances))
        relaxed = analyze_record(self.record, self.mask, cusps, Config())
        self.assertIsNotNone(relaxed['selected'])
        self.assertFalse(relaxed['ratio_fallback_used'])
        self.assertEqual(relaxed['status'], 'ratio_requires_review')
        self.assertEqual(relaxed['isthmuses'], self.record['isthmuses'])

    def test_existing_ratio_is_not_replaced_by_fallback(self):
        # A single complete requested pair crossing the cavity needs no swap.
        cusps = self.cusps.copy()
        cusps[130:, :90] = False
        strict = analyze_record(self.record, self.mask, cusps, Config(), enable_ratio_fallback=False)
        relaxed = analyze_record(self.record, self.mask, cusps, Config())
        self.assertIsNotNone(strict['selected'])
        self.assertEqual(strict['selected'], relaxed['selected'])
        self.assertFalse(relaxed['ratio_fallback_used'])

    def test_changed_cavity_boundary_is_rejected(self):
        changed = self.mask.copy()
        changed[:, 100:] = False
        with self.assertRaises(ValueError):
            analyze_record(self.record, changed, self.cusps, Config())

    def test_middle_panel_has_only_green_mask_and_red_widths(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = analyze_record(self.record, self.mask, self.cusps, Config())
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "source.png"
            Image.fromarray(np.zeros(self.mask.shape, dtype=np.uint8)).save(image)
            with patch("matplotlib.pyplot.close"):
                save_overlay(result, image, Path(directory) / "ratio.png", "same-size", self.cusps)
                fig = plt.gcf()
                middle = fig.axes[1]
                self.assertEqual(len(middle.lines), len(self.record["isthmuses"]))
                self.assertTrue(all(line.get_color() == "#e00000" for line in middle.lines))
                pixels = np.asarray(middle.images[0].get_array())
                np.testing.assert_allclose(pixels[0, 0], [1, 1, 1])
                np.testing.assert_allclose(pixels[120, 90], np.array([46, 175, 80]) / 255)
            plt.close(fig)

    def test_third_panel_keeps_connections_without_ratio_candidates(self):
        import matplotlib.pyplot as plt

        result = analyze_record(self.record, self.mask, self.cusps, Config())
        result.update(selected=None, candidates=[])
        with tempfile.TemporaryDirectory() as directory:
            with patch("matplotlib.pyplot.close"):
                save_overlay(result, None, Path(directory) / "ratio.png", "same-size", self.cusps)
                fig = plt.gcf()
                third = fig.axes[2]
                connections = [line for line in third.lines if len(line.get_xdata()) == 2]
                self.assertEqual(len(connections), 2)
                self.assertTrue(all("display only" in line.get_label() for line in connections))
                self.assertEqual({line.get_label().split(':')[0] for line in connections}, {'C1-C3', 'C2-C4'})
                endpoints = [np.column_stack((line.get_ydata(), line.get_xdata())) for line in connections]
                self.assertFalse(segments_intersect(*endpoints))
                self.assertTrue(all(line.get_color() == "#0057d9" for line in connections))
                self.assertEqual(result["candidates"], [])
                self.assertIsNone(result["selected"])
            plt.close(fig)

    def test_opposite_side_pairing_propagates_to_distances_and_selected_ratio(self):
        result = analyze_record(self.record, self.mask, self.cusps, Config())
        self.assertTrue(result['anatomy']['opposite_sides_required'])
        self.assertTrue(result['selected']['opposite_sides_valid'])
        self.assertTrue(result['anatomy']['cusp_labels_reassigned'])
        self.assertIn(result['selected']['cusp_pair_label'], ('C1-C3', 'C2-C4'))
        self.assertEqual([r['cusp_pair_label'] for r in distance_rows(result)], ['C1-C3', 'C2-C4'])
        self.assertEqual(result['isthmuses'], self.record['isthmuses'])

    def test_disjoint_completion_displays_both_lines_but_excludes_same_side_ratio(self):
        import matplotlib.pyplot as plt

        rows, cols = np.mgrid[:240, :240]
        cavity = (rows >= 90) & (rows <= 160) & (abs(cols-125) <= 9)
        cusps = np.zeros_like(cavity)
        for r, c in [[55, 155], [80, 155], [190, 80], [190, 155]]:
            cusps |= ((rows-r)/10)**2+((cols-c)/8)**2 <= 1
        record = detect_isthmuses(cavity, cusp_mask=cusps)
        record.update(filename='disjoint_mask.png', mask_shape=list(cavity.shape))
        result = analyze_record(record, cavity, cusps, Config())
        self.assertTrue(result['pair_recovery_used'])
        self.assertEqual(len(result['anatomy']['pairs']), 2)
        pairs = result['anatomy']['pairs']
        self.assertEqual(sorted(i for p in pairs for i in p['region_ids']), [1, 2, 3, 4])
        self.assertFalse(segments_intersect(*(p['centers_rc'] for p in pairs)))
        self.assertTrue(pairs[1]['side_rule_conflict'])
        self.assertTrue(result['selected']['opposite_sides_valid'])
        self.assertTrue(all(row['cusp_pair_id'] != pairs[1]['pair_id'] for row in result['candidates']))
        exported = distance_rows(result)
        self.assertFalse(exported[1]['eligible_for_ratio'])
        self.assertTrue(exported[1]['pair_completed_by_exclusion'])
        with tempfile.TemporaryDirectory() as directory:
            with patch('matplotlib.pyplot.close'):
                save_overlay(result, None, Path(directory)/'disjoint.png', 'same-size', cusps)
                fig = plt.gcf()
                third = fig.axes[2]
                lines = [line for line in third.lines if len(line.get_xdata()) == 2]
                self.assertEqual(len(lines), 2)
                labels = {text.get_text() for text in third.texts}
                self.assertTrue({'C1', 'C2', 'C3', 'C4'}.issubset(labels))
                self.assertNotIn('C5', labels)
            plt.close(fig)


class ExcludedCuspRescueTests(unittest.TestCase):
    def setUp(self):
        rows, cols = np.mgrid[:240, :240]
        self.mask = (rows >= 35) & (rows <= 220) & (cols >= 115) & (cols <= 129)
        self.cusps = np.zeros_like(self.mask)
        for r, c in [(57, 90), (72, 150), (105, 90), (166, 150)]:
            self.cusps |= ((rows-r)/9)**2 + ((cols-c)/8)**2 <= 1
        self.cusps |= (rows-185)**2 + (cols-90)**2 <= 4**2

    def analyze(self, row, **kwargs):
        neck = {"isthmus_id": 1, "component_id": 1, "width_px": 15.,
                "center_rc": [row, 122.], "endpoints_rc": [[row, 114.5], [row, 129.5]]}
        record = {"filename": "five_mask.png", "mask_shape": list(self.mask.shape),
                  "isthmuses": [neck], "warnings": []}
        return analyze_record(record, self.mask, self.cusps, Config(), **kwargs)

    def test_existing_near_pair_keeps_smallest_cusp_excluded(self):
        result = self.analyze(140)
        self.assertTrue(result['selected']['within_near_tolerance'])
        self.assertFalse(result['anatomy'].get('excluded_cusp_rescue_used', False))
        self.assertEqual(result['anatomy']['excluded_region_ids'], [5])

    def test_excluded_cusp_can_supply_strict_near_match_without_relaxation(self):
        result = self.analyze(180, enable_ratio_fallback=False)
        anatomy = result['anatomy']
        self.assertTrue(anatomy['excluded_cusp_rescue_used'])
        self.assertEqual(result['selected']['cusp_pair_label'], 'C4-C5')
        self.assertTrue(result['selected']['within_near_tolerance'])
        self.assertFalse(result['selected']['ratio_is_fallback'])
        self.assertEqual([p['region_ids'] for p in anatomy['pairs']], [[1, 3], [4, 5]])
        self.assertEqual(anatomy['excluded_region_ids'], [2])
        self.assertEqual(anatomy['initial_excluded_region_ids'], [5])
        self.assertFalse(segments_intersect(*(p['centers_rc'] for p in anatomy['pairs'])))
        self.assertEqual(result['isthmuses'][0]['width_px'], 15.)
        self.assertAlmostEqual(result['selected']['ratio_isthmus_to_intercuspal'], 15/np.hypot(19, 60))
        distances = distance_rows(result)
        self.assertTrue(distances[1]['uses_reinstated_cusp'])
        self.assertEqual(distances[1]['reinstated_cusp_ids'], '[5]')

    def test_far_improved_pair_remains_explicitly_outside_proximity(self):
        result = self.analyze(210)
        selected = result['selected']
        self.assertEqual(selected['cusp_pair_label'], 'C4-C5')
        self.assertFalse(selected['within_near_tolerance'])
        self.assertTrue(selected['ratio_is_fallback'])
        self.assertEqual(selected['ratio_fallback_stage'], 'proximity_relaxed')
        self.assertFalse(selected['crossing_checks_relaxed'])
        audit = result['excluded_cusp_rescue_audit']
        self.assertLess(audit['match_distance_px'], audit['baseline_gap_px'])

    def test_no_ratio_fallback_does_not_accept_far_replacement(self):
        result = self.analyze(210, policy='near-only', enable_ratio_fallback=False)
        self.assertIsNone(result['selected'])
        self.assertFalse(result['anatomy'].get('excluded_cusp_rescue_used', False))
        self.assertEqual(result['anatomy']['excluded_region_ids'], [5])

    def test_replacement_must_improve_gap_if_not_strictly_near(self):
        result = self.analyze(80)
        self.assertFalse(result['selected']['within_near_tolerance'])
        self.assertFalse(result['anatomy'].get('excluded_cusp_rescue_used', False))
        self.assertEqual(result['anatomy']['excluded_region_ids'], [5])

    def test_crossing_replacement_is_rejected_while_other_pair_is_preserved(self):
        result = self.analyze(180)
        trials = result['excluded_cusp_rescue_audit']['trials']
        rejected = next(t for t in trials if t['region_ids'] == [3, 5])
        self.assertEqual(rejected['status'], 'rejected_crossing_or_reused_cusp')
        used = [i for p in result['anatomy']['pairs'] for i in p['region_ids']]
        self.assertEqual(len(used), len(set(used)))

    def test_reinstatement_requires_valid_cavity_crossing(self):
        def no_crossings(mask, pairs, config):
            return [{'pair_id': p['pair_id'], 'status': 'no_cavity_between_cusps'} for p in pairs]
        with patch('cus_ist_ratio.cusp_guided_sites', side_effect=no_crossings):
            result = self.analyze(180)
        self.assertFalse(result['anatomy'].get('excluded_cusp_rescue_used', False))
        self.assertEqual(result['anatomy']['excluded_region_ids'], [5])


if __name__ == "__main__":
    unittest.main()
