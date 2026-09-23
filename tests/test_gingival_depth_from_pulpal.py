"""Run: python3 -m unittest discover -s tests -p test_gingival_depth_from_pulpal.py."""
import unittest

import numpy as np

import gingival_depth_from_pulpal as depth


def rectangle(x0, x1, y0, y1, height, subdivisions=8):
    faces = []
    x, y = np.linspace(x0, x1, subdivisions+1), np.linspace(y0, y1, subdivisions+1)
    for a, b in zip(x[:-1], x[1:]):
        for c, d in zip(y[:-1], y[1:]):
            p = np.array([[a, c, height(a, c)], [b, c, height(b, c)],
                          [b, d, height(b, d)], [a, d, height(a, d)]])
            faces.extend([p[[0, 1, 2]], p[[0, 2, 3]]])
    return np.array(faces)


def surfaces(gap=2., slope=.0, subdivisions=8):
    return {"occlusal_triangles": rectangle(-4, 0, -2, 2, lambda x, y: slope*x+5, subdivisions),
            "proximal_triangles": rectangle(.2, 2, -1, 1, lambda x, y: slope*x+5-gap, subdivisions)}


class RelativeDepthTests(unittest.TestCase):
    def test_camera_expands_for_depth_marker_outside_floor_bounds(self):
        focus = np.zeros(3)
        eye = np.array([0., -1., 0.])
        self.assertAlmostEqual(depth.marker_camera_scale(focus, 1., eye, [[0, 0, .5]], 1.), 1.)
        self.assertGreater(depth.marker_camera_scale(focus, 1., eye, [[0, 0, 2.]], 1.), 2.)

    def test_displayed_pulpal_depth_preserves_existing_offset_once(self):
        data = {"occlusal_points": np.array([[0., 0., 1.], [1., 0., 1.]]),
                "occlusal_weights": np.array([1., 1.]),
                "occlusal_raw_depths": np.array([2., 2.5]),
                "occlusal_depths": np.array([1.25, 1.75]),
                "occlusal_depth_offset_mm": np.array(.75)}
        result = {"metrics": {}}
        depth.attach_pulpal_measurements(result, data, {"OcclusalDepthMedian_mm": "1.5", "OcclusalStatus": "measured"})
        self.assertAlmostEqual(result["metrics"]["PulpalDepthMedian_mm"], 1.5)
        self.assertAlmostEqual(result["metrics"]["PulpalDepthUnoffsetMedian_mm"], 2.25)
        np.testing.assert_allclose(result["pulpal_arrow_hit"]-result["pulpal_arrow_point"], [0., 0., 1.25])

    def test_unoffset_pulpal_median_uses_raw_samples_after_clipping(self):
        data = {"occlusal_points": np.zeros((2, 3)), "occlusal_weights": np.array([1., 1.]),
                "occlusal_raw_depths": np.array([.25, .5]), "occlusal_depths": np.array([0., 0.]),
                "occlusal_depth_offset_mm": np.array(.75)}
        result = {"metrics": {}}
        depth.attach_pulpal_measurements(result, data, {"OcclusalDepthMedian_mm": "0", "OcclusalStatus": "measured"})
        self.assertAlmostEqual(result["metrics"]["PulpalDepthMedian_mm"], 0.)
        self.assertAlmostEqual(result["metrics"]["PulpalDepthUnoffsetMedian_mm"], .375)

    def test_horizontal_floor_known_gap(self):
        r = depth.measure(surfaces(2.))
        np.testing.assert_allclose(r["depths"], 2., atol=1e-10)
        self.assertAlmostEqual(r["metrics"][depth.PREFIX+"Median_mm"], 2.)

    def test_tilted_plane_measures_axial_not_shortest_distance(self):
        r = depth.measure(surfaces(2., slope=.5))
        np.testing.assert_allclose(r["depths"], 2., atol=1e-9)
        self.assertNotAlmostEqual(r["metrics"][depth.PREFIX+"Median_mm"], 2/np.sqrt(1.25))

    def test_very_deep_student_preparation_is_not_capped(self):
        r = depth.measure(surfaces(8.7))
        np.testing.assert_allclose(r["depths"], 8.7, atol=1e-9)

    def test_negative_student_floor_relationship_is_not_clipped(self):
        r = depth.measure(surfaces(-.8))
        np.testing.assert_allclose(r["depths"], -.8, atol=1e-9)
        self.assertEqual(r["metrics"][depth.PREFIX+"NegativeAreaFraction"], 1.)
        self.assertNotEqual(r["metrics"][depth.PREFIX+"Status"], "unresolved")

    def test_zero_depth_is_valid(self):
        r = depth.measure(surfaces(0.))
        np.testing.assert_allclose(r["depths"], 0., atol=1e-9)

    def test_saved_crown_reference_and_offsets_cannot_change_relative_depth(self):
        data = surfaces()
        expected = depth.measure(data)["depths"]
        data.update(reference_coefficients=np.array([100., -3., 12., 0., 0., 900.]),
                    occlusal_depth_offset_mm=np.array(.75), proximal_depth_offset_mm=np.array(19.))
        np.testing.assert_allclose(depth.measure(data)["depths"], expected, atol=1e-10)

    def test_rigid_rotation_and_axis_rotation_preserve_measurement(self):
        data = surfaces(1.35, slope=.4)
        angle = .87
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        moved = {key: value @ rotation.T + [100., -20., 70.] for key, value in data.items()}
        actual = depth.measure(moved, axis=rotation @ [0., 0., 1.])
        np.testing.assert_allclose(actual["depths"], 1.35, atol=1e-9)

    def test_signed_depth_reverses_with_axis(self):
        plane = {"center": np.array([0., 0., 3.]), "normal": np.array([0., 0., 1.])}
        self.assertAlmostEqual(depth.signed_depth([[1., 2., 1.]], plane, [0., 0., -1.])[0], -2.)

    def test_area_weights_prevent_mesh_density_bias(self):
        data = surfaces()
        large = rectangle(.2, 3.2, -1.5, 1.5, lambda x, y: 4., 2)
        small = rectangle(3.3, 4.3, -.5, .5, lambda x, y: 0., 25)
        data["proximal_triangles"] = np.concatenate([large, small])
        r = depth.measure(data)
        self.assertAlmostEqual(r["metrics"][depth.PREFIX+"Median_mm"], 1., places=8)
        self.assertAlmostEqual(r["metrics"][depth.PREFIX+"Mean_mm"], 1.4, places=8)

    def test_reference_choice_depends_on_position_not_measured_depth(self):
        a, b = depth.measure(surfaces(.05)), depth.measure(surfaces(9.))
        np.testing.assert_array_equal(a["selected_face_mask"], b["selected_face_mask"])
        np.testing.assert_allclose(a["plane"]["center"], b["plane"]["center"])

    def test_tessellation_does_not_change_constant_depth(self):
        for count in [2, 7, 13]:
            r = depth.measure(surfaces(1.75, slope=.31, subdivisions=count))
            self.assertAlmostEqual(r["metrics"][depth.PREFIX+"Median_mm"], 1.75, places=8)

    def test_point_arrow_intersects_plane_and_stays_on_axis(self):
        r = depth.measure(surfaces(1.4, slope=.3))
        delta = r["arrow_hit"]-r["arrow_point"]
        np.testing.assert_allclose(delta, r["arrow_depth_mm"]*r["frame"][2], atol=1e-10)
        self.assertAlmostEqual((r["arrow_hit"]-r["plane"]["center"]) @ r["plane"]["normal"], 0.)

    def test_plane_parallel_to_axis_is_explicitly_unresolved(self):
        plane = {"center": np.zeros(3), "normal": np.array([1., 0., 0.])}
        with self.assertRaisesRegex(ValueError, "parallel"):
            depth.signed_depth([[0., 0., 0.]], plane, [0., 0., 1.])

    def test_degenerate_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "collinear"):
            depth.fit_plane(np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]), np.ones(3))

    def test_robust_fit_limits_isolated_reference_bumps(self):
        xx, yy = np.meshgrid(np.linspace(-3., 3., 31), np.linspace(-3., 3., 31))
        p = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)*4])
        p[:30, 2] += 1.
        ordinary = depth.fit_plane(p, np.ones(len(p)), robust=False)
        robust = depth.fit_plane(p, np.ones(len(p)))
        robust_error = abs(depth.signed_depth([[0., 0., 0.]], robust, [0., 0., 1.])[0]-4)
        ordinary_error = abs(depth.signed_depth([[0., 0., 0.]], ordinary, [0., 0., 1.])[0]-4)
        self.assertLess(robust_error, ordinary_error/10)

    def test_nonfinite_axis_and_zero_area_are_rejected(self):
        for axis in [[0, 0, 0], [0, np.nan, 1]]:
            with self.assertRaises(ValueError):
                depth.axis_frame(axis)
        with self.assertRaisesRegex(ValueError, "zero-area"):
            depth.quadrature(np.zeros((1, 3, 3)))


if __name__ == "__main__":
    unittest.main()
