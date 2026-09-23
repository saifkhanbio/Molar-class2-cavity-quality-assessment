"""Run with python3 -m unittest discover -s tests -p 'test_*cav_con_comb_smooth.py'."""
import unittest

import numpy as np
from PIL import Image, ImageDraw

import landmark_cav_con_comb_smooth as cavity


class ImageTransformTests(unittest.TestCase):
    def test_small_prediction_islands_are_excluded(self):
        mask = np.zeros((20, 20), bool)
        mask[2:6, 2:6] = True
        mask[15:17, 15:17] = True
        selected, count, discarded = cavity.largest_mask_component(mask)
        self.assertEqual(count, 2)
        self.assertEqual(selected.sum(), 16)
        self.assertAlmostEqual(discarded, .2)

    def test_resize_maps_pixel_centers_and_composes(self):
        first = cavity.resize_matrix((1430, 565), (256, 256))
        second = cavity.resize_matrix((256, 256), (128, 128))
        np.testing.assert_allclose(second @ first, cavity.resize_matrix((1430, 565), (128, 128)))
        center = np.array([(1430 - 1) / 2, (565 - 1) / 2, 1])
        np.testing.assert_allclose(first @ center, [127.5, 127.5, 1])

    def test_warp_uses_forward_xy_transform(self):
        source = np.zeros((20, 20))
        source[5, 7] = 1
        transform = np.array([[1, 0, 3], [0, 1, -2], [0, 0, 1]])
        actual = cavity.warp_image(source, transform, source.shape, order=0, background=0)
        self.assertEqual(actual[3, 10], 1)
        self.assertEqual(actual.sum(), 1)

    def test_composed_camera_maps_to_prediction_grid(self):
        camera = np.array([[20, 0, 5, 30], [0, 20, 5, 10], [0, 0, .1, 1.]])
        transform = np.array([[0, -1, 30], [1, 0, 4], [0, 0, 1.]])
        point = np.array([1., 2, 3, 1])
        projected = camera @ point
        expected = transform @ (projected / projected[2])
        actual = (transform @ camera) @ point
        np.testing.assert_allclose(actual[:2] / actual[2], expected[:2])

    def test_affine_recovers_rotated_asymmetric_rendering(self):
        image = Image.new("RGB", (128, 128), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((36, 12, 89, 116), fill=(160, 160, 160))
        draw.ellipse((48, 23, 67, 62), fill=(65, 65, 65))
        draw.rectangle((66, 65, 77, 92), fill=(105, 105, 105))
        angle = np.radians(23)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        matrix = np.eye(3)
        matrix[:2, :2] = rotation
        matrix[:2, 2] = np.array([66, 61]) - rotation @ [64, 64]
        moved = cavity.warp_image(np.asarray(image.convert("L")), matrix, (128, 128))
        estimate, info = cavity.fit_image_affine(image, Image.fromarray(np.uint8(moved)).convert("RGB"))
        landmarks = np.array([[54, 40, 1], [72, 84, 1], [62, 100, 1]])
        error = np.linalg.norm((landmarks @ estimate.T - landmarks @ matrix.T)[:, :2], axis=1)
        self.assertLess(error.max(), 1.)
        self.assertGreater(info["image_affine_correlation"], .97)


class FloorSelectionTests(unittest.TestCase):
    def geometry(self, n):
        geometry = {"centers": np.column_stack([np.arange(n) * .1, np.zeros(n), np.ones(n)]),
                    "area": np.ones(n), "pairs": np.column_stack([np.arange(n - 1), np.arange(1, n)])}
        fields = {"edge": np.linspace(3, .5, n), "depth": np.ones(n) * 2,
                  "nz": np.ones(n), "crown": np.ones(n, bool), "concavity": np.ones(n)}
        return geometry, fields

    def candidate(self, ids, score, edge):
        return {"ids": np.array(ids), "score": score, "area_mm2": len(ids),
                "depth_median_mm": 2., "edge_distance_mm": edge, "concave_fraction": 1.}

    def guidance(self, occlusal, proximal):
        return {name: ({"use_for_selection": True}, np.array(values, bool), None, None)
                for name, values in [("Occlusal", occlusal), ("Proximal", proximal)]}

    def test_distinct_proximal_box_survives_overlapping_view(self):
        geometry, fields = self.geometry(3)
        candidates = [self.candidate([0, 1], 10, 3), self.candidate([2], 2, .5)]
        guidance = self.guidance([1, 1, 0], [0, 1, 1])
        o, p, _ = cavity.select_landmark_floors(candidates, geometry, fields, guidance)
        np.testing.assert_array_equal(o["ids"], [0, 1])
        np.testing.assert_array_equal(p["ids"], [2])

    def test_shared_floor_partition_rejects_far_occlusal_support(self):
        geometry, fields = self.geometry(4)
        fields["edge"] = np.array([3., 3., .8, .5])
        candidates = [self.candidate([0, 1, 2, 3], 10, 2)]
        guidance = self.guidance([1, 1, 1, 1], [1, 0, 1, 0])
        o, p, _ = cavity.select_landmark_floors(candidates, geometry, fields, guidance)
        self.assertIn(0, o["ids"])
        self.assertNotIn(0, p["ids"])
        self.assertFalse(np.intersect1d(o["ids"], p["ids"]).size)

    def test_steep_rim_does_not_replace_shared_proximal_floor(self):
        geometry, fields = self.geometry(4)
        fields["nz"] = np.array([1., 1., 1., .49])
        fields["edge"] = np.array([3., 2., .8, .5])
        candidates = [self.candidate([0, 1, 2], 10, 2), self.candidate([3], 2, .5)]
        guidance = self.guidance([1, 1, 1, 0], [0, 0, 1, 1])
        _, p, _ = cavity.select_landmark_floors(candidates, geometry, fields, guidance)
        np.testing.assert_array_equal(p["ids"], [2])

    def test_shared_floor_does_not_assign_opposite_opening_to_proximal(self):
        geometry, fields = self.geometry(4)
        geometry["centers"][:, 0] = [0., 3., 4., 5.]
        fields["edge"] = np.array([.5, 3., 2., .5])
        candidates = [self.candidate([0, 1, 2, 3], 10, 2)]
        guidance = self.guidance([1, 1, 1, 1], [1, 0, 0, 1])
        guidance["Proximal"][0]["toward_camera"] = [1., 0., 0.]
        o, p, _ = cavity.select_landmark_floors(candidates, geometry, fields, guidance)
        self.assertIn(0, o["ids"])
        self.assertNotIn(0, p["ids"])
        self.assertIn(3, p["ids"])

    def test_terminal_partition_fills_side_view_occlusion_gap(self):
        geometry, fields = self.geometry(4)
        geometry["centers"][:, 0] = [0., .5, 1., 1.5]
        fields["edge"] = np.array([3., 2., .8, .5])
        candidates = [self.candidate([0, 1, 2, 3], 10, 2)]
        guidance = self.guidance([1, 1, 1, 1], [0, 1, 0, 1])
        guidance["Proximal"][0]["toward_camera"] = [1., 0., 0.]
        _, p, _ = cavity.select_landmark_floors(candidates, geometry, fields, guidance)
        np.testing.assert_array_equal(p["ids"], [1, 2, 3])

    def test_occlusal_identity_survives_stronger_proximal_mask_coverage(self):
        geometry, fields = self.geometry(5)
        candidates = [self.candidate([0, 1], 10, 3), self.candidate([2, 3, 4], 2, .5)]
        guidance = self.guidance([1, 0, 1, 1, 1], [0, 0, 1, 1, 1])
        o, p, _ = cavity.select_landmark_floors(candidates, geometry, fields, guidance)
        self.assertTrue(np.isin(o["ids"], [0, 1]).all())
        np.testing.assert_array_equal(p["ids"], [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
