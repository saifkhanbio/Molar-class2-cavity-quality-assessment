"""Run: python3 -m unittest discover -s tests -p test_publication_cavity_depth.py."""
import unittest

import numpy as np

import publication_cavity_depth as publication


def measurement_data(pulpal=(2., 2., 2.), gingival=(4., 4., 4.), weights=None):
    data = {"reference_coefficients": np.array([0., 0., 0., 0., 0., 10.])}
    for region, depths in (("occlusal", pulpal), ("proximal", gingival)):
        depths = np.asarray(depths)
        points = np.column_stack([np.arange(len(depths)), np.zeros(len(depths)), 10-depths])
        plane_center = np.array([0., 0., points[:, 2].mean()])
        plane_normal = np.array([0., 0., 1.])
        prefix = region+"_"
        data.update({prefix+"points": points, prefix+"weights": np.ones(len(depths)) if weights is None else np.array(weights),
                     prefix+"raw_depths": depths, prefix+"depths": np.maximum(depths-.75, 0) if region == "occlusal" else depths.copy(),
                     prefix+"depth_offset_mm": np.array(.75 if region == "occlusal" else 0.),
                     prefix+"triangles": np.repeat(points[:, None, :], 3, axis=1),
                     prefix+"plane_center": plane_center, prefix+"plane_normal": plane_normal,
                     prefix+"residuals": (points-plane_center) @ plane_normal})
    return data


class CommonReferenceTests(unittest.TestCase):
    def test_interactive_cameras_match_preview_directions(self):
        for view, eye in (("overview", publication.OVERVIEW_EYE),
                          ("top", publication.DETAIL_EYES["occlusal"]),
                          ("proximal", publication.DETAIL_EYES["proximal"])):
            camera = publication.preview_camera(view)
            actual = np.array([camera["eye"][axis] for axis in "xyz"])
            up = np.array([camera["up"][axis] for axis in "xyz"])
            np.testing.assert_allclose(actual/np.linalg.norm(actual), eye/np.linalg.norm(eye), atol=1e-12)
            self.assertAlmostEqual(float(actual @ up), 0.)
            self.assertAlmostEqual(float(np.linalg.norm(up)), 1.)
            self.assertEqual(camera["projection"]["type"], "orthographic")

    def test_known_parallel_floor_depths(self):
        metrics, _ = publication.common_reference_metrics(measurement_data())
        self.assertEqual(metrics["OcclusalDepthMedian_mm"], 1.25)
        self.assertEqual(metrics["ProximalDepthMedian_mm"], 3.25)
        self.assertEqual(metrics["GingivalMinusPulpalDepth_mm"], 2.)
        self.assertNotIn("OcclusalDepthOffset_mm", metrics)
        self.assertNotIn("ProximalDepthOffset_mm", metrics)

    def test_clipped_source_cannot_be_restored_by_adding_offset(self):
        data = measurement_data(pulpal=(.25, .5, .6))
        self.assertTrue(np.all(data["occlusal_depths"] == 0))
        metrics, samples = publication.common_reference_metrics(data)
        self.assertAlmostEqual(metrics["OcclusalDepthMedian_mm"], -.25)
        np.testing.assert_allclose(samples["occlusal"]["depths"], [-.5, -.25, -.15])

    def test_negative_difference_is_preserved(self):
        metrics, _ = publication.common_reference_metrics(measurement_data(pulpal=(5.,)*3))
        self.assertEqual(metrics["GingivalMinusPulpalDepth_mm"], -1.)

    def test_signed_reference_depth_is_not_clipped(self):
        metrics, samples = publication.common_reference_metrics(measurement_data(pulpal=(-.5,)*3))
        self.assertEqual(metrics["OcclusalDepthMedian_mm"], -1.25)
        self.assertTrue(np.all(samples["occlusal"]["depths"] < 0))

    def test_area_weights_affect_medians_and_rms(self):
        metrics, _ = publication.common_reference_metrics(
            measurement_data(pulpal=(1., 2., 8.), weights=(1., 8., 1.)))
        self.assertAlmostEqual(metrics["OcclusalDepthMedian_mm"], 1.25)
        expected_rms = np.sqrt(np.average((np.array([1., 2., 8.])-11/3)**2, weights=[1., 8., 1.]))
        self.assertAlmostEqual(metrics["OcclusalFloorResidualRMS_mm"], expected_rms)

    def test_difference_of_medians_not_median_of_arbitrarily_paired_samples(self):
        metrics, _ = publication.common_reference_metrics(
            measurement_data(pulpal=(1., 2., 10.), gingival=(10., 3., 4.)))
        self.assertEqual(metrics["GingivalMinusPulpalDepth_mm"], 2.)
        self.assertNotEqual(metrics["GingivalMinusPulpalDepth_mm"], np.median([9., 1., -6.]))

    def test_saved_arrays_are_not_modified(self):
        data = measurement_data()
        copies = {key: value.copy() for key, value in data.items()}
        publication.common_reference_metrics(data)
        for key in copies:
            np.testing.assert_array_equal(data[key], copies[key])

    def test_both_floors_have_same_reduction_for_all_depth_statistics(self):
        data = measurement_data(pulpal=(1., 2., 8.), gingival=(4., 5., 6.))
        metrics, samples = publication.common_reference_metrics(data)
        for region in ("occlusal", "proximal"):
            raw = data[region+"_raw_depths"]
            np.testing.assert_allclose(samples[region]["depths"], raw-.75)
            for suffix, value in (("Mean", raw.mean()), ("Median", np.median(raw)),
                                  ("Min", raw.min()), ("Max", raw.max()), ("P95", raw.max())):
                self.assertAlmostEqual(metrics[region.title()+"Depth"+suffix+"_mm"], value-.75)
        self.assertEqual(metrics["GingivalMinusPulpalDepth_mm"], 3.)

    def test_reference_marker_uses_same_surface_as_both_measured_depths(self):
        data = measurement_data(pulpal=(1., 2., 8.), gingival=(4., 5., 6.))
        _, samples = publication.common_reference_metrics(data)
        for region in ("occlusal", "proximal"):
            points = data[region+"_points"]
            np.testing.assert_allclose(points[:, 2]+samples[region]["depths"], 9.25)
            np.testing.assert_allclose(publication.publication_reference_z(points, data["reference_coefficients"]), 9.25)

    def test_existing_case_view_preferences_are_preserved(self):
        self.assertEqual(publication.case_camera({"stem": "O_37"}, initial=True), publication.preview_camera())
        self.assertEqual(publication.case_camera({"stem": "O_1"}, initial=True)["eye"],
                         {"x": 1.05, "y": -1.65, "z": 1.65})

    def test_invalid_area_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            publication.common_reference_metrics(measurement_data(weights=(0., 0., 0.)))

    def test_mismatched_raw_reference_values_are_rejected(self):
        data = measurement_data()
        data["occlusal_raw_depths"] += .1
        with self.assertRaises(AssertionError):
            publication.common_reference_metrics(data)


if __name__ == "__main__":
    unittest.main()
