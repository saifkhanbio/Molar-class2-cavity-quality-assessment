"""Geometric checks: python3 -m unittest discover -s tests -p 'test_final_cav_con_comb_smooth.py' -v."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import final_cav_con_comb_smooth as cavity


def surface(function, n=12):
    x, y = np.meshgrid(np.linspace(-1, 1, n + 1), np.linspace(-1, 1, n + 1))
    points = np.stack([x, y, function(x, y)], axis=-1)
    return np.array([tri for i in range(n) for j in range(n)
                     for tri in [[points[i, j], points[i + 1, j], points[i, j + 1]],
                                 [points[i + 1, j], points[i + 1, j + 1], points[i, j + 1]]]])


class FloorMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.array([0, 0, 0, 0, 0, 4.0])

    def measure(self, triangles):
        return cavity.measure_floor(triangles, self.reference)

    def test_flat_floor_has_correct_unoffset_depth_and_zero_roughness(self):
        result = self.measure(surface(lambda x, y: np.full_like(x, 2.0)))
        for name in ['DepthMean_mm', 'DepthMedian_mm', 'DepthP95_mm', 'DepthMax_mm']:
            self.assertAlmostEqual(result['metrics'][name], 2.0, places=10)
        self.assertLess(result['metrics']['FloorResidualRMS_mm'], 1e-10)
        self.assertAlmostEqual(result['metrics']['FloorArea_mm2'], 4, places=10)

    def test_occlusal_offset_changes_all_depths_but_not_smoothness(self):
        triangles = surface(lambda x, y: 2 + 0.1 * x ** 2)
        raw = self.measure(triangles)
        corrected = cavity.measure_floor(triangles, self.reference, cavity.DEPTH_OUTPUT_OFFSET_MM)
        for key in ['DepthMean_mm', 'DepthMedian_mm', 'DepthP95_mm', 'DepthMax_mm',
                    'DepthMin_mm', 'ApproxNormalDepthMedian_mm']:
            self.assertAlmostEqual(corrected['metrics'][key], raw['metrics'][key] - 0.75)
        for key in ['FloorResidualRMS_mm', 'FloorResidualMeanAbs_mm', 'FloorResidualAbsP95_mm', 'FloorTilt_deg']:
            self.assertEqual(corrected['metrics'][key], raw['metrics'][key])
        np.testing.assert_array_equal(corrected['raw_depths'], raw['depths'])
        np.testing.assert_allclose(corrected['depths'], raw['depths'] - 0.75)
        for a, b in zip(corrected['regions'], raw['regions']):
            for key in ['DepthMedian_mm', 'DepthP95_mm']:
                self.assertAlmostEqual(a[key], b[key] - 0.75)

    def test_occlusal_offset_clamps_individual_depths_before_aggregation(self):
        triangles = surface(lambda x, y: 3.4 + 0.5 * x)
        result = cavity.measure_floor(triangles, self.reference, 0.75)
        expected = np.maximum(result['raw_depths'] - 0.75, 0)
        np.testing.assert_array_equal(result['depths'], expected)
        self.assertEqual(result['metrics']['DepthMin_mm'], 0)
        self.assertAlmostEqual(result['metrics']['DepthMean_mm'], np.average(expected, weights=result['weights']))
        self.assertGreater(result['metrics']['DepthClippedAreaFraction'], 0)

    def test_invalid_offset_is_rejected(self):
        for value in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                cavity.measure_floor(surface(lambda x, y: x * 0), self.reference, value)

    def test_tilted_floor_is_smooth_but_has_depth_variation(self):
        result = self.measure(surface(lambda x, y: 2 + 0.2 * x + 0.1 * y))
        self.assertLess(result['metrics']['FloorResidualRMS_mm'], 1e-9)
        self.assertAlmostEqual(result['metrics']['FloorTilt_deg'],
                               np.degrees(np.arctan(np.sqrt(0.05))), places=6)
        self.assertGreater(result['metrics']['RegionalDepthMedianSD_mm'], 0)
        self.assertAlmostEqual(result['metrics']['DepthMax_mm'], 2.3, places=9)

    def test_rough_surface_scores_worse_than_plane_and_scales_with_amplitude(self):
        mild = self.measure(surface(lambda x, y: 2 + 0.03 * np.cos(3 * np.pi * x) * np.cos(3 * np.pi * y)))
        rough = self.measure(surface(lambda x, y: 2 + 0.12 * np.cos(3 * np.pi * x) * np.cos(3 * np.pi * y)))
        self.assertGreater(mild['metrics']['FloorResidualRMS_mm'], 0.005)
        self.assertGreater(rough['metrics']['FloorResidualRMS_mm'],
                           mild['metrics']['FloorResidualRMS_mm'] * 3)

    def test_subdividing_only_one_side_does_not_bias_surface_area_or_rms(self):
        original = surface(lambda x, y: 2 + 0.1 * x ** 2 + 0.08 * y ** 2)
        left = original.mean(axis=1)[:, 0] < 0
        refined = np.concatenate([cavity.subdivide(original[left], max_edge=0.04), original[~left]])
        a, b = self.measure(original)['metrics'], self.measure(refined)['metrics']
        self.assertAlmostEqual(a['FloorArea_mm2'], b['FloorArea_mm2'], places=9)
        self.assertAlmostEqual(a['FloorResidualRMS_mm'], b['FloorResidualRMS_mm'], delta=0.0002)
        self.assertAlmostEqual(a['DepthMean_mm'], b['DepthMean_mm'], places=9)

    def test_roughness_is_invariant_to_rigid_pose_change(self):
        points = surface(lambda x, y: 2 + 0.1 * x ** 2).reshape(-1, 3)
        weights = np.ones(len(points))
        _, _, r1 = cavity.fit_floor_plane(points, weights)
        t = 0.3
        rotation = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        _, _, r2 = cavity.fit_floor_plane(points @ rotation.T + [9, -12, 5], weights)
        self.assertAlmostEqual(np.mean(r1 ** 2), np.mean(r2 ** 2), places=10)

    def test_walls_and_duplicate_faces_are_excluded(self):
        floor = surface(lambda x, y: np.full_like(x, 2.0))
        wall = np.array([[[0, 0, 2], [0, 0, 3], [0, 1, 2]]], float)
        all_faces = np.concatenate([floor, floor[:, ::-1], wall])
        selected = cavity.select_measurement_triangles(all_faces, all_faces.reshape(-1, 3), 0.8)
        self.assertEqual(len(selected), len(floor))
        self.assertAlmostEqual(self.measure(selected)['metrics']['FloorArea_mm2'], 4, places=9)

    def test_missing_support_is_missing_not_zero(self):
        self.assertIsNone(self.measure(np.empty((0, 3, 3))))
        self.assertIsNone(self.measure(surface(lambda x, y: x * 0, n=1)))
        self.assertTrue(np.isnan(cavity.weighted_quantile([], [], 0.5)))

    def test_negative_depths_are_preserved_and_flagged(self):
        result = self.measure(surface(lambda x, y: np.full_like(x, 4.2)))
        self.assertAlmostEqual(result['metrics']['DepthMedian_mm'], -0.2)
        self.assertEqual(result['metrics']['NegativeDepthAreaFraction'], 1.0)

    def test_vertex_maximum_is_not_mean_of_regional_maxima(self):
        result = self.measure(surface(lambda x, y: 2 - 0.3 * y))
        self.assertAlmostEqual(result['metrics']['DepthMax_mm'], 2.3, places=9)
        self.assertLess(result['metrics']['DepthMedian_mm'], result['metrics']['DepthMax_mm'])

    def test_inclined_reference_distinguishes_vertical_and_approx_normal_depth(self):
        triangles = surface(lambda x, y: 2 + 0.5 * x)
        result = cavity.measure_floor(triangles, np.array([0, 0, 0, 0.5, 0, 4.0]))
        self.assertAlmostEqual(result['metrics']['DepthMedian_mm'], 2, places=10)
        self.assertAlmostEqual(result['metrics']['ApproxNormalDepthMedian_mm'], 2 / np.sqrt(1.25), places=10)

    def test_plot_legend_uses_reported_statistics_including_vertex_max(self):
        result = self.measure(surface(lambda x, y: 2 - 0.3 * y))
        label = cavity.depth_legend('Occlusal', result['depths'], result['metrics'])
        self.assertIn('max 2.30 mm', label)
        self.assertIn('median 2.00', label)
        self.assertNotIn('sample', label)

    def test_empty_regions_remain_missing(self):
        left = surface(lambda x, y: x * 0 + 2)
        right = left + np.array([0, 10, 0])
        result = self.measure(np.concatenate([left, right]))
        self.assertTrue(np.isnan(result['regions'][1]['DepthMedian_mm']))
        self.assertTrue(np.isnan(result['metrics']['RegionalDepthMedianSD_mm']))

    def test_degenerate_plane_is_rejected(self):
        with self.assertRaises(ValueError):
            cavity.fit_floor_plane(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), np.ones(3))

    def test_csv_and_json_missing_values_are_not_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cavity.write_json(path / 'result.json', {'value': float('nan')})
            cavity.write_table(path / 'result.csv', [{'value': float('nan')}])
            self.assertIn('null', (path / 'result.json').read_text())
            self.assertNotIn('nan', (path / 'result.csv').read_text())


if __name__ == '__main__':
    unittest.main()
