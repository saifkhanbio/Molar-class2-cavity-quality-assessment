"""Geometry and I/O regression tests: python3 -m unittest discover -s tests -v."""

import tempfile
from dataclasses import replace
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from isthmus_disc import (Config, analyze_cusps, csv_row, detect_isthmuses,
                          filter_intercuspal_sites, map_to_image, read_mask,
                          save_figure, segment_distance, serializable)


def detect_strict(mask, config=None, cusp_mask=None):
    """Exercise the conservative detector independently of fallback selection."""
    return detect_isthmuses(mask, replace(config or Config(), enable_fallback=False), cusp_mask)


def passage(necks=(), curved=False):
    rows, cols = np.mgrid[:240, :180]
    widths = np.full(240, 36.0)
    for center in necks:
        widths -= 20 * np.exp(-0.5 * ((np.arange(240) - center) / 10) ** 2)
    centers = 90 + (12 * np.sin(np.arange(240) / 55) if curved else np.zeros(240))
    return (rows >= 20) & (rows <= 219) & (np.abs(cols - centers[:, None]) < widths[:, None] / 2)


def cusp_regions(connected=False, single_pair=False):
    rows, cols = np.mgrid[:240, :180]
    mask = np.zeros((240, 180), bool)
    for row in ((80,) if single_pair else (80, 160)):
        for col in (45, 135):
            mask |= ((rows - row) / 22) ** 2 + ((cols - col) / 12) ** 2 <= 1
    if connected and not single_pair:
        mask[80:161, 43:48] = True
    return mask


class DetectionTests(unittest.TestCase):
    def test_two_known_necks_and_boundary_widths(self):
        result = detect_isthmuses(passage((80, 160)))
        self.assertEqual(result["status"], "two_isthmuses")
        for candidate, expected_row in zip(result["isthmuses"], (80, 160)):
            self.assertAlmostEqual(candidate["center_rc"][0], expected_row, delta=2)
            self.assertAlmostEqual(candidate["center_rc"][1], 90, delta=1)
            self.assertAlmostEqual(candidate["width_px"], 16, delta=1.5)
            self.assertIsNone(candidate["width_mm"])
            a, b = np.asarray(candidate["endpoints_rc"])
            self.assertAlmostEqual(np.linalg.norm(a - b), candidate["width_px"])
            np.testing.assert_allclose((a + b) / 2, candidate["center_rc"])

    def test_one_neck_is_not_reported_twice(self):
        result = detect_isthmuses(passage((120,)))
        self.assertEqual(len(result["isthmuses"]), 1)
        self.assertAlmostEqual(result["isthmuses"][0]["center_rc"][0], 120, delta=2)

    def test_broad_flat_neck_uses_both_valley_shoulders(self):
        rows, cols = np.mgrid[:240, :140]
        widths = np.interp(np.arange(240),
                           [0, 20, 40, 60, 100, 140, 180, 200, 220, 239],
                           [0, 3, 17, 17, 11, 11, 23, 23, 3, 0])
        mask = (rows >= 20) & (rows < 220) & (np.abs(cols - 70) < widths[:, None] / 2)
        result = detect_isthmuses(mask)
        self.assertEqual(result["status"], "one_isthmus")
        candidate = result["isthmuses"][0]
        self.assertAlmostEqual(candidate["center_rc"][0], 120, delta=8)
        self.assertAlmostEqual(candidate["width_px"], 11, delta=1)
        low, high = np.asarray(candidate["near_minimum_interval_rc"])[:, 0]
        self.assertGreater(abs(high - low), 20)
        self.assertGreater(candidate["relative_prominence"], 0.25)

    def test_uniform_passage_and_taper_have_no_necks(self):
        uniform = detect_strict(passage())
        self.assertEqual(uniform["isthmuses"], [])
        self.assertTrue(uniform["review_reasons"])
        rows, cols = np.mgrid[:180, :120]
        taper = (rows > 20) & (rows < 160) & (np.abs(cols - 60) < (10 + rows * 0.08))
        self.assertEqual(detect_strict(taper)["isthmuses"], [])

    def test_curved_passage_retains_two_necks(self):
        result = detect_isthmuses(passage((80, 160), curved=True))
        self.assertEqual(len(result["isthmuses"]), 2)
        for candidate, row in zip(result["isthmuses"], (80, 160)):
            self.assertAlmostEqual(candidate["center_rc"][0], row, delta=4)
            self.assertAlmostEqual(candidate["width_px"], 16, delta=2)

    def test_rotation_does_not_remove_horizontal_or_diagonal_necks(self):
        for angle in (37, 90, 143):
            with self.subTest(angle=angle):
                rotated = ndi.rotate(passage((80, 160)), angle, reshape=True, order=0)
                result = detect_isthmuses(rotated)
                self.assertEqual(len(result["isthmuses"]), 2)
                for candidate in result["isthmuses"]:
                    self.assertAlmostEqual(candidate["width_px"], 16, delta=2)

    def test_uniform_scaling_preserves_locations(self):
        result = detect_isthmuses(ndi.zoom(passage((80, 160)), 2, order=0))
        self.assertEqual(len(result["isthmuses"]), 2)
        for candidate, row in zip(result["isthmuses"], (160, 320)):
            self.assertAlmostEqual(candidate["center_rc"][0], row, delta=5)
            self.assertAlmostEqual(candidate["width_px"], 32, delta=3)

    def test_small_isolated_noise_does_not_change_detection(self):
        mask = passage((80, 160))
        mask[5:7, 5:7] = True
        result = detect_isthmuses(mask)
        self.assertEqual(len(result["isthmuses"]), 2)
        self.assertEqual(result["discarded_foreground_pixels"], 4)

    def test_distinct_components_are_retained(self):
        first = passage((120,))
        mask = np.concatenate([first, first], axis=1)
        result = detect_isthmuses(mask)
        self.assertEqual(len(result["isthmuses"]), 2)
        self.assertEqual(len({c["component_id"] for c in result["isthmuses"]}), 2)
        self.assertIn("multiple_cavity_components", result["warnings"])

    def test_mm_calibration_does_not_change_locations(self):
        plain = detect_isthmuses(passage((120,)))
        calibrated = detect_isthmuses(passage((120,)), Config(pixel_size_mm=0.05))
        a, b = plain["isthmuses"][0], calibrated["isthmuses"][0]
        np.testing.assert_array_equal(a["center_rc"], b["center_rc"])
        self.assertAlmostEqual(b["width_mm"], b["width_px"] * 0.05)

    def test_empty_tiny_and_closed_ring_masks(self):
        self.assertEqual(detect_isthmuses(np.zeros((40, 40), bool))["status"], "empty_mask")
        tiny = np.zeros((40, 40), bool)
        tiny[20, 20] = True
        self.assertEqual(detect_isthmuses(tiny)["isthmuses"], [])
        y, x = np.mgrid[:120, :120]
        radius = np.hypot(x - 60, y - 60)
        self.assertEqual(detect_isthmuses((radius > 25) & (radius < 35))["isthmuses"], [])


class CuspSupportTests(unittest.TestCase):
    def test_connected_cusps_are_split_without_forcing_a_count(self):
        anatomy = analyze_cusps(cusp_regions(connected=True), passage())
        self.assertEqual(len(anatomy["regions"]), 4)
        self.assertEqual(len(anatomy["pairs"]), 2)
        self.assertEqual(anatomy["split_connected_regions"], 1)
        for pair in anatomy["pairs"]:
            centers = np.asarray(pair["centers_rc"])
            self.assertLess(abs(centers[0, 0] - centers[1, 0]), 2)
            self.assertAlmostEqual(pair["intercusp_distance_px"], 90, delta=1)

    def test_uniform_cavity_sites_remain_separate_from_neck_detections(self):
        result = detect_strict(passage(), cusp_mask=cusp_regions())
        self.assertEqual(result["isthmuses"], [])
        available = [site for site in result["cusp_guided_sites"] if site["status"] == "available"]
        self.assertEqual(len(available), 2)
        for site, row in zip(available, (80, 160)):
            self.assertEqual(site["kind"], "cusp_guided_estimate")
            self.assertAlmostEqual(site["center_rc"][0], row, delta=1)
            self.assertAlmostEqual(site["width_px"], 35, delta=1)
            self.assertIsNone(site["associated_isthmus_id"])
            self.assertEqual(site["interpretation"], "possible_isthmus")
            self.assertTrue(site["draw_intercuspal_line"])
        self.assertEqual(csv_row({**result, "filename": "uniform.png"})["n_isthmuses"], 0)

    def test_cusp_pairs_support_real_necks(self):
        result = detect_strict(passage((80, 160)), cusp_mask=cusp_regions(connected=True))
        self.assertEqual(len(result["isthmuses"]), 2)
        self.assertEqual({candidate["cusp_pair_id"] for candidate in result["isthmuses"]}, {1, 2})
        self.assertTrue(all(candidate["cusp_support"] for candidate in result["isthmuses"]))
        self.assertEqual({site["associated_isthmus_id"] for site in result["cusp_guided_sites"]}, {1, 2})

    def test_one_opposing_pair_is_not_forced_into_two(self):
        result = detect_strict(passage(), cusp_mask=cusp_regions(single_pair=True))
        self.assertEqual(len(result["cusp_anatomy"]["pairs"]), 1)
        self.assertEqual(len(result["cusp_guided_sites"]), 1)
        self.assertEqual(result["cusp_guided_sites"][0]["status"], "available")

    def test_cusp_line_with_no_cavity_is_unavailable(self):
        mask = passage()
        mask[:120] = False
        result = detect_strict(mask, cusp_mask=cusp_regions())
        self.assertEqual(result["cusp_guided_sites"][0]["status"], "no_cavity_between_cusps")
        self.assertIsNone(result["cusp_guided_sites"][0]["width_px"])
        self.assertFalse(result["cusp_guided_sites"][0]["draw_intercuspal_line"])

    def test_existing_neck_drops_distant_second_line_without_fallback(self):
        result = detect_strict(passage((80,)), cusp_mask=cusp_regions())
        self.assertEqual(len(result["isthmuses"]), 1)
        first, second = result["cusp_guided_sites"]
        self.assertTrue(first["draw_intercuspal_line"])
        self.assertEqual(first["interpretation"], "near_detected_isthmus")
        self.assertEqual(first["associated_isthmus_id"], 1)
        self.assertFalse(second["draw_intercuspal_line"])
        self.assertEqual(second["status"], "dropped_not_near_detected_isthmus")
        self.assertIsNone(second["width_px"])
        self.assertIn("rejected_crossing", second)

    def test_distant_crossings_neither_draw_nor_support_detected_necks(self):
        result = detect_strict(passage((100, 180)), cusp_mask=cusp_regions())
        self.assertEqual(len(result["isthmuses"]), 2)
        self.assertEqual(len(result["cusp_guided_sites"]), 2)
        self.assertTrue(all(not site["draw_intercuspal_line"] for site in result["cusp_guided_sites"]))
        self.assertTrue(all(not neck["cusp_support"] for neck in result["isthmuses"]))

    def test_nearby_crossings_within_width_tolerance_are_retained(self):
        result = detect_strict(passage((83, 163)), cusp_mask=cusp_regions())
        self.assertEqual(len(result["isthmuses"]), 2)
        self.assertEqual(len(result["cusp_guided_sites"]), 2)
        for site in result["cusp_guided_sites"]:
            self.assertTrue(site["draw_intercuspal_line"])
            self.assertGreater(site["nearest_isthmus_distance_px"], 0)
            self.assertLessEqual(site["nearest_isthmus_distance_px"], site["near_threshold_px"])

    def test_crossing_in_other_component_is_not_supported_by_near_neck(self):
        site = {"status": "available", "component_id": 2, "center_rc": [10, 5],
                "endpoints_rc": [[10, 0], [10, 10]], "width_px": 10, "width_mm": None}
        neck = {"component_id": 1, "endpoints_rc": [[11, 0], [11, 10]],
                "isthmus_id": 1, "width_px": 10}
        result = filter_intercuspal_sites([site], [neck], Config())[0]
        self.assertFalse(result["draw_intercuspal_line"])
        self.assertEqual(result["status"], "dropped_not_near_detected_isthmus")

    def test_removed_noise_preserves_crossing_component_association(self):
        mask = passage((80, 160))
        mask[0, 0] = True
        result = detect_strict(mask, cusp_mask=cusp_regions())
        self.assertEqual(len(result["cusp_guided_sites"]), 2)
        self.assertTrue(all(site["draw_intercuspal_line"] for site in result["cusp_guided_sites"]))
        self.assertTrue(all(neck["cusp_support"] for neck in result["isthmuses"]))

    def test_finite_segment_distance(self):
        self.assertEqual(segment_distance([[0, 0], [0, 10]], [[-2, 5], [2, 5]]), 0)
        self.assertEqual(segment_distance([[0, 0], [0, 10]], [[3, 0], [3, 10]]), 3)
        self.assertEqual(segment_distance([[0, 0], [0, 10]], [[0, 5], [0, 15]]), 0)
        # Infinite extensions intersect, but the finite segments do not.
        self.assertEqual(segment_distance([[0, 0], [0, 10]], [[2, 15], [5, 15]]), np.sqrt(29))
        self.assertEqual(segment_distance([[0, 0], [0, 0]], [[3, 0], [3, 10]]), 3)

    def test_figures_draw_only_eligible_intercuspal_lines(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tooth.png"
            Image.fromarray(np.zeros((240, 180), dtype=np.uint8)).save(source)
            for necks, expected in (((80,), 1), ((100, 180), 0), ((), 2)):
                with self.subTest(necks=necks):
                    result = detect_strict(passage(necks), cusp_mask=cusp_regions())
                    result["filename"] = "synthetic_mask.png"
                    map_to_image(result, (240, 180), (180, 240), "same-size")
                    with patch("matplotlib.pyplot.close"):
                        save_figure(result, source, Path(directory) / "overlay.png")
                        figure = plt.gcf()
                        for panel, axis in enumerate(figure.axes[:2]):
                            displayed = [line for line in axis.lines
                                         if line.get_color() == "#4899dc" and line.get_linestyle() == ":"]
                            self.assertEqual(len(displayed), expected if panel == 0 else 0)
                        middle = figure.axes[1]
                        self.assertEqual(len(middle.lines), len(result["isthmuses"]))
                        self.assertTrue(all(line.get_color() == "#e00000" for line in middle.lines))
                        pixels = np.asarray(middle.images[0].get_array())
                        np.testing.assert_allclose(pixels[0, 0], [1, 1, 1])
                        np.testing.assert_allclose(pixels[120, 90], np.array([46, 175, 80]) / 255)
                    plt.close(figure)

    def test_multiple_crossings_are_not_measured_across_background(self):
        mask = passage()
        mask[:, 88:93] = False
        result = detect_strict(mask, cusp_mask=cusp_regions())
        self.assertEqual(len(result["cusp_guided_sites"]), 2)
        self.assertTrue(all(site["status"] == "ambiguous_multiple_cavity_crossings"
                            for site in result["cusp_guided_sites"]))
        self.assertTrue(all(site["width_px"] is None for site in result["cusp_guided_sites"]))

    def test_cusp_site_calibration_and_image_mapping(self):
        result = detect_strict(passage(), Config(pixel_size_mm=0.02), cusp_regions())
        self.assertEqual(len(result["cusp_guided_sites"]), 2)
        map_to_image(result, (240, 180), (360, 480), "resize")
        for site in result["cusp_guided_sites"]:
            self.assertEqual(site["status"], "available")
            self.assertAlmostEqual(site["width_mm"], site["width_px"] * 0.02)
            self.assertAlmostEqual(site["width_image_px"], site["width_px"] * 2)
            np.testing.assert_allclose(site["image_endpoints_rc"],
                                       (np.array(site["endpoints_rc"]) + 0.5) * 2 - 0.5)

    def test_cusp_support_rotates_with_the_images(self):
        for angle in (37, 90):
            with self.subTest(angle=angle):
                mask = ndi.rotate(passage((80, 160)), angle, reshape=True, order=0)
                cusps = ndi.rotate(cusp_regions(connected=True), angle, reshape=True, order=0)
                result = detect_strict(mask, cusp_mask=cusps)
                self.assertEqual(len(result["cusp_anatomy"]["pairs"]), 2)
                self.assertEqual(len(result["isthmuses"]), 2)
                self.assertTrue(all(candidate["cusp_support"] for candidate in result["isthmuses"]))

    def test_empty_or_unaligned_cusps_do_not_change_geometry(self):
        mask = passage((120,))
        original = detect_strict(mask)["isthmuses"][0]
        for cusp_mask in (np.zeros_like(mask), np.zeros((20, 20), bool)):
            result = detect_strict(mask, cusp_mask=cusp_mask)
            self.assertEqual(len(result["isthmuses"]), 1)
            np.testing.assert_allclose(result["isthmuses"][0]["center_rc"], original["center_rc"])
            self.assertEqual(result["cusp_guided_sites"], [])


class FallbackTests(unittest.TestCase):
    def test_uniform_cavity_gets_one_interior_estimate(self):
        result = detect_isthmuses(passage())
        self.assertEqual(result["status"], "one_isthmus_fallback")
        self.assertEqual(len(result["isthmuses"]), 1)
        candidate = result["isthmuses"][0]
        self.assertEqual(candidate["detection_method"], "fallback_interior_section")
        self.assertTrue(candidate["is_fallback"])
        self.assertEqual(candidate["confidence_label"], "low")
        self.assertAlmostEqual(candidate["center_rc"][0], 120, delta=10)
        self.assertAlmostEqual(candidate["width_px"], 35, delta=1)
        self.assertGreaterEqual(candidate["cut_fraction"], 0.10)
        self.assertEqual(result["fallback_audit"][-1]["outcome"], candidate["detection_method"])
        row = csv_row({**result, "filename": "uniform.png"})
        self.assertEqual(row["n_fallback_isthmuses"], 1)
        self.assertEqual(row["n_geometric_isthmuses"], 0)

    def test_fallback_does_not_change_existing_necks(self):
        for necks in ((120,), (80, 160)):
            standard = detect_strict(passage(necks))
            result = detect_isthmuses(passage(necks))
            self.assertFalse(result["fallback_used"])
            self.assertEqual(result["fallback_audit"], [])
            self.assertEqual(len(result["isthmuses"]), len(standard["isthmuses"]))
            for current, previous in zip(result["isthmuses"], standard["isthmuses"]):
                np.testing.assert_allclose(current["endpoints_rc"], previous["endpoints_rc"])

    def test_weak_valley_uses_relaxed_thresholds(self):
        rows, cols = np.mgrid[:240, :180]
        widths = 36 - 3 * np.exp(-0.5 * ((np.arange(240) - 120) / 12) ** 2)
        widths *= np.sqrt(np.maximum(0, 1 - ((np.arange(240) - 120) / 100) ** 8))
        mask = np.abs(cols - 90) < widths[:, None] / 2
        self.assertFalse(detect_strict(mask)["isthmuses"])
        result = detect_isthmuses(mask)
        self.assertEqual(result["isthmuses"][0]["detection_method"], "fallback_relaxed_valley")
        self.assertAlmostEqual(result["isthmuses"][0]["center_rc"][0], 120, delta=5)

    def test_cusp_crossing_promoted_to_one_labeled_fallback(self):
        result = detect_isthmuses(passage(), cusp_mask=cusp_regions())
        self.assertEqual(len(result["isthmuses"]), 1)
        candidate = result["isthmuses"][0]
        self.assertEqual(candidate["detection_method"], "fallback_cusp_crossing")
        self.assertEqual(candidate["confidence_label"], "low")
        sites = [site for site in result["cusp_guided_sites"] if site["draw_intercuspal_line"]]
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0]["interpretation"], "near_fallback_isthmus")
        np.testing.assert_allclose(sites[0]["endpoints_rc"], candidate["endpoints_rc"])

    def test_fallback_rotation_and_calibration(self):
        original = detect_isthmuses(passage())["isthmuses"][0]
        result = detect_isthmuses(ndi.rotate(passage(), 90, reshape=True, order=0), Config(pixel_size_mm=0.02))
        self.assertEqual(len(result["isthmuses"]), 1)
        candidate = result["isthmuses"][0]
        self.assertAlmostEqual(candidate["width_px"], original["width_px"], delta=1)
        self.assertAlmostEqual(candidate["width_mm"], candidate["width_px"] * 0.02)

    def test_empty_and_noise_only_masks_remain_unmeasurable(self):
        empty = np.zeros((40, 40), bool)
        for mask in (empty, np.eye(40, dtype=bool) & (np.indices((40, 40))[0] < 2)):
            result = detect_isthmuses(mask)
            self.assertEqual(result["isthmuses"], [])
            self.assertFalse(result["fallback_used"])


class OutputTests(unittest.TestCase):
    def test_unverified_resize_does_not_invent_image_coordinates(self):
        result = detect_isthmuses(passage((120,)))
        map_to_image(result, (240, 180), (900, 480), "same-size")
        self.assertEqual(result["image_mapping"], "unverified_size_mismatch")
        self.assertNotIn("image_center_rc", result["isthmuses"][0])

    def test_explicit_resize_maps_pixel_centers_and_endpoints(self):
        result = detect_isthmuses(passage((120,)))
        map_to_image(result, (240, 180), (900, 480), "resize")
        candidate = result["isthmuses"][0]
        np.testing.assert_allclose(candidate["image_center_rc"],
                                   (np.array(candidate["center_rc"]) + 0.5) * [2, 5] - 0.5)
        self.assertAlmostEqual(candidate["width_image_px"], candidate["width_px"] * 5, delta=0.1)

    def test_missing_measurements_are_blank_not_zero(self):
        result = detect_isthmuses(passage((120,)))
        result["filename"] = "example_mask.png"
        row = csv_row(result)
        self.assertIsNone(row["isthmus2_width_px"])
        self.assertIsNone(row["isthmus2_center_row"])
        self.assertIsNone(row["isthmus1_width_mm"])

    def test_png_binary_encodings_and_finite_json(self):
        with tempfile.TemporaryDirectory() as folder:
            mask = passage((120,))
            for maximum in (1, 255):
                path = Path(folder) / f"mask_{maximum}.png"
                Image.fromarray(mask.astype(np.uint8) * maximum).save(path)
                np.testing.assert_array_equal(read_mask(path), mask)
        self.assertEqual(serializable(np.array([np.nan, np.inf, 1.0])), [None, None, 1.0])


if __name__ == "__main__":
    unittest.main()
