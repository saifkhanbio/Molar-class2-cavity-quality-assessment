"""Run with python3 -m unittest discover -s tests -v."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from final_avg_efd_score import (align_descriptors, efd_coefficients, extract_descriptor,
                                 parse_args, read_binary_mask, score_masks)


class DatasetPresetTests(unittest.TestCase):
    def parse(self, argv):
        with patch.object(Path, 'is_dir', return_value=True):
            return parse_args(argv)

    def test_default_selects_matching_occlusal_inputs(self):
        args = self.parse([])
        self.assertEqual(args.cavity_type, 'occlusal')
        self.assertEqual(args.pred_folder, Path('pred_O_M_masks_folder'))
        self.assertEqual(args.ref_folder, Path('Scoring_ref_O_M_mask'))
        self.assertEqual(args.samples_folder, Path('samples_stu_O'))
        self.assertEqual(args.output_dir, Path('final_avg_efd_results/occlusal'))

    def test_proximal_preset_and_folder_inference_select_same_inputs(self):
        explicit = self.parse(['--cavity-type', 'proximal'])
        inferred = self.parse(['--pred-folder', 'pred_P_M_masks_folder'])
        self.assertEqual(vars(explicit), vars(inferred))
        self.assertEqual(explicit.ref_folder, Path('Scoring_ref_P_M_mask'))
        self.assertEqual(explicit.samples_folder, Path('samples_stu_P'))
        self.assertEqual(explicit.output_dir, Path('final_avg_efd_results/proximal'))

    def test_recognized_mixed_datasets_are_rejected(self):
        cases = [ ['--cavity-type', 'proximal', '--pred-folder', 'pred_O_M_masks_folder'],
                  ['--ref-folder', 'Scoring_ref_P_M_mask'], ['--samples-folder', 'samples_stu_P'] ]
        for argv in cases:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(argv)


class EFDGeometryTests(unittest.TestCase):
    def setUp(self):
        self.contour = np.array([[0, 0], [5, 0], [4, 1], [6, 4], [2, 6], [-1, 4]], float)
        self.descriptor = efd_coefficients(self.contour, 18)

    def score(self, contour):
        return align_descriptors(self.descriptor, efd_coefficients(contour, 18))['similarity_pct']

    def test_identical_contour_is_100_percent(self):
        self.assertEqual(self.score(self.contour), 100.0)

    def test_start_point_and_winding_do_not_change_score(self):
        for shift in range(len(self.contour)):
            for points in (self.contour, self.contour[::-1]):
                self.assertGreater(self.score(np.roll(points, shift, axis=0)), 99.9999)

    def test_translation_rotation_and_uniform_scale_do_not_change_score(self):
        for angle in (0.23, 1.7, 3.1):
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            for scale in (0.2, 5):
                self.assertGreater(self.score(scale * self.contour @ rotation + [123, -87]), 99.9999)

    def test_closure_repeated_points_and_segment_subdivision_do_not_change_score(self):
        closed = np.vstack((self.contour, self.contour[0]))
        repeated = np.repeat(closed, 2, axis=0)
        subdivided = np.array([p + t * (q - p) for p, q in zip(self.contour, np.roll(self.contour, -1, axis=0))
                              for t in (0, 0.1, 0.3, 0.8)])
        for points in (closed, repeated, subdivided):
            self.assertGreater(self.score(points), 99.9999)

    def test_aspect_ratio_and_shape_changes_are_not_normalized_away(self):
        self.assertLess(self.score(self.contour * [3, 1]), 85)
        self.assertLess(self.score(np.array([[0, 0], [6, 0], [3, 6]])), 95)

    def test_symmetry_and_bounded_scores(self):
        b = efd_coefficients(self.contour * [1.5, 1])
        ab, ba = align_descriptors(self.descriptor, b), align_descriptors(b, self.descriptor)
        self.assertAlmostEqual(ab['distance'], ba['distance'], places=7)
        self.assertGreaterEqual(ab['similarity_pct'], 0)
        self.assertLessEqual(ab['similarity_pct'], 100)

    def test_circle_has_no_unstable_first_axis_requirement(self):
        t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        circle = np.column_stack((np.cos(t), np.sin(t)))
        a, b = efd_coefficients(circle), efd_coefficients(np.roll(circle * 12 + 5, 23, axis=0))
        self.assertGreater(align_descriptors(a, b)['similarity_pct'], 99.9999)

    def test_invalid_contours_and_orders_are_rejected(self):
        for points in (np.zeros((10, 2)), [[0, 0], [1, 1], [2, 2]], [[0, 0], [1, np.nan], [2, 1]]):
            with self.assertRaises(ValueError):
                efd_coefficients(points)
        for order in (0, -2, 1.5):
            with self.assertRaises(ValueError):
                efd_coefficients(self.contour, order)

    def test_descriptor_dimension_is_four_coefficients_per_harmonic(self):
        for order in (6, 18, 30):
            descriptor = efd_coefficients(self.contour, order)
            self.assertEqual(descriptor.shape, (order, 2, 2))
            self.assertAlmostEqual(float(np.linalg.norm(descriptor)), 1)

    def test_integrated_coefficients_match_independent_dense_fourier_quadrature(self):
        closed = np.vstack((self.contour, self.contour[0]))
        knots = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
        positions = np.arange(65536) / 65536 * knots[-1]
        samples = np.column_stack([np.interp(positions, knots, closed[:, axis]) for axis in (0, 1)])
        spectrum = np.fft.rfft(samples, axis=0)[1:19] * (2 / len(samples))
        numerical = np.stack((spectrum.real, -spectrum.imag), axis=-1)
        numerical /= np.linalg.norm(numerical)
        np.testing.assert_allclose(self.descriptor, numerical, atol=1e-7, rtol=0)


class MaskPipelineTests(unittest.TestCase):
    def setUp(self):
        rows, cols = np.mgrid[:80, :80]
        self.mask = ((rows - 40) / 22) ** 2 + ((cols - 40) / 12) ** 2 <= 1

    def save(self, folder, name, mask, level=255):
        path = Path(folder) / name
        Image.fromarray(mask.astype(np.uint8) * level).save(path)
        return path

    def test_binary_encodings_match(self):
        with tempfile.TemporaryDirectory() as folder:
            a = self.save(folder, 'a.png', self.mask, 1)
            b = self.save(folder, 'b.png', self.mask, 255)
            np.testing.assert_array_equal(read_binary_mask(a), read_binary_mask(b))
            aa, bb = extract_descriptor(a), extract_descriptor(b)
            self.assertEqual(aa['mask_hash'], bb['mask_hash'])
            self.assertEqual(align_descriptors(aa['coefficients'], bb['coefficients'])['similarity_pct'], 100)

    def test_small_secondary_component_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            noisy = self.mask.copy(); noisy[2:4, 2:4] = True
            item = extract_descriptor(self.save(folder, 'noise.png', noisy))
            clean = extract_descriptor(self.save(folder, 'clean.png', self.mask))
            self.assertEqual(item['discarded_area_px'], 4)
            self.assertIn('secondary_components_not_scored', item['warnings'])
            self.assertEqual(align_descriptors(item['coefficients'], clean['coefficients'])['similarity_pct'], 100)

    def test_empty_mask_rejected_and_border_contour_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                extract_descriptor(self.save(folder, 'empty.png', np.zeros_like(self.mask)))
            border = np.zeros_like(self.mask); border[:30, :20] = True
            item = extract_descriptor(self.save(folder, 'border.png', border))
            self.assertTrue(np.isfinite(item['coefficients']).all())
            self.assertIn('foreground_touches_image_boundary', item['warnings'])

    def test_pipeline_separates_exact_match_from_mean_and_deduplicates_refs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); pred = root / 'pred'; ref = root / 'ref'; out = root / 'out'
            pred.mkdir(); ref.mkdir()
            self.save(pred, 'P_1_mask.png', self.mask)
            self.save(pred, 'P_11_mask.png', self.mask)
            self.save(pred, 'empty.png', np.zeros_like(self.mask))
            self.save(ref, 'P_1.png', self.mask)
            self.save(ref, 'P_11.png', self.mask)
            square = np.zeros_like(self.mask); square[20:60, 20:60] = True
            self.save(ref, 'square.png', square)
            with contextlib.redirect_stdout(io.StringIO()):
                details = score_masks(pred, ref, root / 'missing_samples', 18, out, plots=False)
            first = details['results'][0]
            self.assertEqual(first['similarity_pct'], 100)
            self.assertLess(first['average_matching_score_pct'], 99)
            self.assertGreater(first['average_all_files_pct'], first['average_matching_score_pct'])
            self.assertEqual(first['unique_reference_count'], 2)
            self.assertEqual(details['exact_duplicate_checks'][0]['similarity_pct'], 100)
            self.assertEqual(details['results'][-1]['status'], 'INVALID_MASK')
            self.assertEqual(json.loads((out / 'details.json').read_text())['efd_order'], 18)
            self.assertTrue((out / 'reference_comparisons.csv').exists())

    def test_supplied_O1_O11_duplicate_regression(self):
        root = Path(__file__).resolve().parents[1]
        for folder, suffix in [('pred_O_M_masks_folder', '_mask.png'), ('Scoring_ref_O_M_mask', '.png')]:
            first, second = root / folder / ('O_1' + suffix), root / folder / ('O_11' + suffix)
            if not first.exists() or not second.exists():
                self.skipTest('Sample masks are not installed.')
            a, b = extract_descriptor(first), extract_descriptor(second)
            self.assertEqual(a['mask_hash'], b['mask_hash'])
            self.assertEqual(align_descriptors(a['coefficients'], b['coefficients'])['similarity_pct'], 100)


if __name__ == '__main__':
    unittest.main()
