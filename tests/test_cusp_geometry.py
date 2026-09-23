"""Geometry regressions for merged lobes, rotations and incomplete masks."""
import unittest

import numpy as np
from scipy import ndimage as ndi

from cusp_geometry import (analyze_cusp_geometry, opposite_side_pairs, pairing_plan,
                           complete_disjoint_pairs, requested_pairs,
                           validate_pair_set, segments_intersect,
                           largest_four_regions, split_touching_lobes)


class NonCrossingPairTests(unittest.TestCase):
    def regions(self, points):
        return [{'region_id': i, 'center_rc': list(point)} for i, point in enumerate(points, 1)]

    def test_crossed_defaults_switch_both_pairs_without_moving_centers(self):
        regions = self.regions([[20, 20], [20, 100], [100, 100], [100, 20]])
        before = [dict(r) for r in regions]
        plan = pairing_plan(regions)
        pairs, unused = requested_pairs(regions)
        self.assertTrue(plan['pairing_switched_for_crossing'])
        self.assertEqual([p['region_ids'] for p in pairs], [[1, 4], [2, 3]])
        self.assertFalse(segments_intersect(pairs[0]['centers_rc'], pairs[1]['centers_rc']))
        self.assertEqual(regions, before)
        self.assertEqual(unused, [])

    def test_non_crossing_defaults_are_preserved(self):
        regions = self.regions([[20, 20], [20, 100], [100, 20], [100, 100]])
        self.assertFalse(pairing_plan(regions)['pairing_switched_for_crossing'])
        pairs, _ = requested_pairs(regions)
        self.assertEqual([p['region_ids'] for p in pairs], [[1, 3], [2, 4]])

    def test_crossing_decision_survives_rotation_scaling_and_translation(self):
        points = np.array([[20, 20], [20, 100], [100, 100], [100, 20]])
        for angle in (0, 31, 90, 143):
            theta = np.deg2rad(angle)
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            for scale in (0.2, 1, 5):
                transformed = scale * points @ rotation.T + [123, -456]
                pairs, _ = requested_pairs(self.regions(transformed))
                self.assertEqual([p['region_ids'] for p in pairs], [[1, 4], [2, 3]])
                self.assertFalse(segments_intersect(pairs[0]['centers_rc'], pairs[1]['centers_rc']))

    def test_infinite_line_intersection_does_not_count_outside_segments(self):
        self.assertFalse(segments_intersect([[0, 0], [0, 10]], [[5, 5], [10, 5]]))

    def test_overlap_and_touch_are_detected_but_disjoint_collinear_lines_are_not(self):
        self.assertTrue(segments_intersect([[0, 0], [0, 10]], [[0, 5], [0, 15]]))
        self.assertTrue(segments_intersect([[0, 0], [0, 10]], [[0, 10], [5, 10]]))
        self.assertFalse(segments_intersect([[0, 0], [0, 10]], [[0, 11], [0, 15]]))

    def test_degenerate_alternative_never_draws_two_overlapping_lines(self):
        regions = self.regions([[0, 0], [0, 10], [0, 30], [0, 20]])
        plan = pairing_plan(regions)
        pairs, _ = requested_pairs(regions)
        self.assertTrue(plan['pairing_degenerate'])
        self.assertEqual([p['region_ids'] for p in pairs], [[1, 4]])


class CuspGeometryTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.cols = np.mgrid[:240, :240]

    def ellipse(self, row, col, height=32, width=18):
        return ((self.rows - row) / height) ** 2 + ((self.cols - col) / width) ** 2 <= 1

    def analyze(self, mask):
        cavity = (abs(self.cols-118) <= 2) & (self.rows > 20) & (self.rows < 220)
        return analyze_cusp_geometry(mask, cavity)

    def test_broadly_overlapping_lobes_have_separate_centers(self):
        mask = self.ellipse(70, 105) | self.ellipse(70, 130)
        before = mask.copy()
        self.assertEqual(ndi.label(mask)[1], 1)
        result = self.analyze(mask)
        self.assertEqual(len(result['regions']), 2)
        centers = np.array([r['center_rc'] for r in result['regions']])
        self.assertLess(centers[:, 1].min(), 115)
        self.assertGreater(centers[:, 1].max(), 120)
        self.assertTrue(result['split_audit'])
        np.testing.assert_array_equal(mask, before)

    def test_convex_single_cusp_is_not_forced_into_two(self):
        mask = self.ellipse(110, 110)
        separated, audit = split_touching_lobes(mask)
        np.testing.assert_array_equal(separated, mask)
        self.assertEqual(audit, [])
        self.assertEqual(len(self.analyze(mask)['regions']), 1)

    def test_two_merged_banks_produce_four_cores_two_disjoint_pairs(self):
        mask = np.zeros_like(self.rows, dtype=bool)
        for row in (65, 175):
            for col in (105, 130):
                mask |= self.ellipse(row, col)
        result = self.analyze(mask)
        self.assertEqual(len(result['regions']), 4)
        self.assertEqual(len(result['pairs']), 2)
        ids = [i for p in result['pairs'] for i in p['region_ids']]
        self.assertEqual(len(set(ids)), 4)

    def test_rotation_preserves_cores_and_only_uses_requested_ids(self):
        mask = np.zeros_like(self.rows, dtype=bool)
        for row in (65, 175):
            for col in (105, 130):
                mask |= self.ellipse(row, col)
        reference = self.analyze(mask)
        for angle in (31, 90, 143):
            rotated = ndi.rotate(mask.astype(float), angle, reshape=False, order=0) > 0.5
            cavity = (abs(self.cols-118) <= 2) & (self.rows > 20) & (self.rows < 220)
            cavity = ndi.rotate(cavity.astype(float), angle, reshape=False, order=0) > .5
            result = analyze_cusp_geometry(rotated, cavity)
            self.assertEqual(len(result['regions']), 4)
            self.assertEqual(len(result['pairs']), 2)
            self.assertEqual([p['region_ids'] for p in result['pairs']], [[1, 3], [2, 4]])
            # Detection-order IDs can change under rotation. Compare the two
            # within-merged-region separations independently of those IDs.
            from scipy.spatial.distance import pdist
            reference_centers = [region['center_rc'] for region in reference['regions']]
            centers = [region['center_rc'] for region in result['regions']]
            np.testing.assert_allclose(np.sort(pdist(centers))[:2],
                                       np.sort(pdist(reference_centers))[:2], atol=4)


    def test_small_isolated_cusp_is_retained_but_speckles_are_not(self):
        mask = self.ellipse(65, 65) | self.ellipse(175, 65) | self.ellipse(65, 145)
        mask |= self.ellipse(175, 145, 10, 4)
        mask[10:12, 10:12] = True
        result = self.analyze(mask)
        self.assertEqual(len(result['regions']), 4)
        self.assertEqual(len(result['pairs']), 2)

    def test_three_cores_do_not_invent_a_fourth_or_reuse_one(self):
        mask = self.ellipse(65, 65) | self.ellipse(175, 65) | self.ellipse(65, 145)
        result = self.analyze(mask)
        self.assertEqual(len(result['regions']), 3)
        self.assertEqual(len(result['pairs']), 1)
        self.assertEqual(len(result['unpaired_region_ids']), 1)

    def test_empty_and_border_masks_are_safe(self):
        self.assertEqual(self.analyze(np.zeros_like(self.rows, bool))['regions'], [])
        result = self.analyze(self.ellipse(20, 0) | self.ellipse(20, 25))
        self.assertTrue(result['regions'])

    def test_requested_pairs_preserve_ids_even_with_missing_and_extra_cores(self):
        points = [[20, 20], [20, 100], [100, 20], [100, 100], [150, 20]]
        regions = [{'region_id': i, 'center_rc': p} for i, p in enumerate(points, 1)]
        pairs, unused = requested_pairs(regions[::-1])
        self.assertEqual([p['region_ids'] for p in pairs], [[1, 3], [2, 4]])
        self.assertEqual(unused, [5])
        pairs, unused = requested_pairs([r for r in regions if r['region_id'] != 3])
        self.assertEqual([p['region_ids'] for p in pairs], [[2, 4]])
        self.assertEqual(pairs[0]['pair_id'], 2)
        self.assertEqual(unused, [1, 5])


class OppositeBankTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.cols = np.mgrid[:260, :260]
        self.cavity = (abs(self.cols-125) <= 8) & (self.rows >= 25) & (self.rows <= 230)

    def regions(self, points):
        return [{"region_id": i, "center_rc": list(p)} for i, p in enumerate(points, 1)]

    def test_row_order_never_pairs_two_cusps_on_same_bank(self):
        regions = self.regions([[65, 85], [65, 165], [185, 85], [185, 165]])
        before = [r['center_rc'][:] for r in regions]
        plan = opposite_side_pairs(regions, self.cavity)
        self.assertEqual([p['region_ids'] for p in plan['pairs']], [[1, 3], [2, 4]])
        self.assertEqual([p['source_region_ids'] for p in plan['pairs']], [[1, 2], [3, 4]])
        self.assertEqual([r['center_rc'] for r in regions], before)
        self.assertFalse(segments_intersect(*(p['centers_rc'] for p in plan['pairs'])))
        self.assertTrue(all(p['opposite_sides_valid'] for p in plan['pairs']))

    def test_all_cusps_on_one_bank_produce_no_pair(self):
        regions = self.regions([[60, 60], [90, 90], [170, 60], [200, 90]])
        self.assertEqual(opposite_side_pairs(regions, self.cavity)['pairs'], [])

    def test_smallest_fifth_core_is_excluded_even_if_geometrically_preferred(self):
        regions = self.regions([[50, 85], [65, 165], [110, 85], [185, 165], [185, 85]])
        for region, area in zip(regions, [1000, 1000, 1000, 1000, 100]):
            region['area_px'] = area
        plan = opposite_side_pairs(regions, self.cavity)
        used = {i for p in plan['pairs'] for i in p['source_region_ids']}
        self.assertEqual(used, {1, 2, 3, 4})
        self.assertEqual(plan['excluded_source_region_ids'], [5])
        self.assertEqual(len(plan['unpaired_region_ids']), 1)

    def test_smallest_core_is_removed_by_area_not_detection_order_or_radius(self):
        regions = self.regions([[50, 85], [65, 165], [110, 85], [185, 165], [185, 85]])
        for region, area in zip(regions, [100, 1000, 1200, 900, 1300]):
            region.update(area_px=area, radius_px=50 if area == 100 else 5)
        plan = opposite_side_pairs(regions, self.cavity)
        used = {i for p in plan['pairs'] for i in p['source_region_ids']}
        self.assertEqual(used, {2, 3, 4, 5})
        self.assertEqual(plan['excluded_source_region_ids'], [1])

    def test_six_detections_drop_both_smallest_before_pairing(self):
        regions = self.regions([[50, 85], [65, 165], [110, 85], [185, 165], [185, 85], [220, 90]])
        for region, area in zip(regions, [50, 1000, 1200, 900, 1300, 75]):
            region['area_px'] = area
        plan = opposite_side_pairs(regions, self.cavity)
        self.assertEqual(plan['excluded_source_region_ids'], [1, 6])
        self.assertEqual(plan['n_pairing_cusp_centers'], 4)
        self.assertEqual({i for p in plan['pairs'] for i in p['source_region_ids']}, {2, 3, 4, 5})

    def test_equal_area_cutoff_is_stable_under_list_reordering(self):
        regions = [{'region_id': i, 'area_px': 100} for i in range(1, 6)]
        for ordered in [regions, regions[::-1], regions[2:]+regions[:2]]:
            selected, excluded = largest_four_regions(ordered)
            self.assertEqual({r['region_id'] for r in selected}, {1, 2, 3, 4})
            self.assertEqual([r['region_id'] for r in excluded], [5])

    def test_four_or_fewer_do_not_lose_their_smallest_core(self):
        regions = [{'region_id': i, 'area_px': i} for i in range(1, 5)]
        selected, excluded = largest_four_regions(regions)
        self.assertEqual(selected, regions)
        self.assertEqual(excluded, [])

    def test_three_versus_one_bank_keeps_only_one_disjoint_pair(self):
        regions = self.regions([[60, 85], [110, 85], [185, 85], [185, 165]])
        plan = opposite_side_pairs(regions, self.cavity)
        self.assertEqual(len(plan['pairs']), 1)
        self.assertEqual(plan['pairs'][0]['source_region_ids'], [3, 4])

    def test_rotation_preserves_physical_pairing(self):
        points = np.array([[65., 85.], [65., 165.], [185., 85.], [185., 165.]])
        for angle in (0, 31, 90, 143):
            theta = np.deg2rad(angle)
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            rotated_points = (points-129.5) @ rotation.T + 129.5
            # scipy rotates row/column coordinates by this same matrix.
            mask = ndi.rotate(self.cavity.astype(float), angle, reshape=False, order=0) > .5
            plan = opposite_side_pairs(self.regions(rotated_points), mask)
            edges = {tuple(sorted(p['source_region_ids'])) for p in plan['pairs']}
            self.assertEqual(edges, {(1, 2), (3, 4)})

    def test_crossing_a_curved_mask_is_not_enough_if_both_cores_on_same_bank(self):
        middle = 80+.007*(self.rows-120)**2
        mask = (abs(self.cols-middle) <= 5) & (self.rows >= 40) & (self.rows <= 200)
        regions = self.regions([[60, 95], [180, 95]])
        self.assertTrue(mask[60:181, 95].any())
        plan = opposite_side_pairs(regions, mask)
        self.assertEqual([r['cavity_side'] for r in regions], [-1, -1])
        self.assertEqual(plan['pairs'], [])

    def test_no_cavity_does_not_invent_bank_evidence(self):
        regions = self.regions([[65, 85], [65, 165], [185, 85], [185, 165]])
        plan = opposite_side_pairs(regions, np.zeros_like(self.cavity))
        self.assertEqual(plan['pairs'], [])
        self.assertEqual(plan['opposite_side_status'], 'no_usable_cavity_axis')


class DisjointCompletionTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.cols = np.mgrid[:240, :240]
        self.mask = (self.rows >= 90) & (self.rows <= 160) & (abs(self.cols-125) <= 9)

    def regions(self, points):
        return [{'region_id': i, 'center_rc': list(p)} for i, p in enumerate(points, 1)]

    def recover(self, points):
        regions = self.regions(points)
        old = opposite_side_pairs(regions, self.mask)
        return regions, complete_disjoint_pairs(regions, self.mask, old)

    def test_shared_cusp_is_rejected_even_when_lines_do_not_cross_interior(self):
        pairs = [{'region_ids': [1, 3], 'centers_rc': [[0, 0], [10, 10]]},
                 {'region_ids': [2, 3], 'centers_rc': [[0, 10], [10, 10]]}]
        with self.assertRaisesRegex(ValueError, 'only one pair'):
            validate_pair_set(pairs)

    def test_distinct_ids_cannot_hide_crossing_or_coincident_centers(self):
        pairs = [{'region_ids': [1, 2], 'centers_rc': [[0, 0], [10, 10]]},
                 {'region_ids': [3, 4], 'centers_rc': [[0, 10], [10, 0]]}]
        with self.assertRaisesRegex(ValueError, 'must not intersect'):
            validate_pair_set(pairs)
        pairs[1]['centers_rc'] = [[0, 0], [10, 0]]
        with self.assertRaisesRegex(ValueError, 'must not intersect'):
            validate_pair_set(pairs)

    def test_first_pair_forces_remaining_two_without_moving_any_center(self):
        points = [[55, 155], [80, 155], [190, 155], [190, 80]]
        regions, recovered = self.recover(points)
        self.assertEqual([r['region_id'] for r in regions], [1, 2, 3, 4])
        self.assertEqual([r['center_rc'] for r in regions], points)
        self.assertEqual([p['region_ids'] for p in recovered['pairs']], [[1, 4], [2, 3]])
        self.assertEqual(recovered['unpaired_region_ids'], [])
        validate_pair_set(recovered['pairs'])
        first, second = recovered['pairs']
        self.assertTrue(first['opposite_sides_valid'])
        self.assertFalse(second['opposite_sides_valid'])
        self.assertTrue(second['pair_completed_by_exclusion'])
        self.assertTrue(second['side_rule_conflict'])
        self.assertFalse(any(p['shared_cusp_pair'] for p in recovered['pairs']))

    def test_first_pair_uses_detected_ids_not_hardcoded_shared_cusp(self):
        _, recovered = self.recover([[55, 155], [80, 155], [190, 80], [190, 155]])
        self.assertEqual([p['region_ids'] for p in recovered['pairs']], [[1, 3], [2, 4]])
        validate_pair_set(recovered['pairs'])

    def test_supported_first_pair_is_preserved_and_complement_added(self):
        _, recovered = self.recover([[55, 155], [80, 155], [140, 80], [140, 160]])
        self.assertEqual([p['region_ids'] for p in recovered['pairs']], [[3, 4], [1, 2]])
        self.assertEqual(recovered['pairs'][1]['pair_role'], 'remaining_two')

    def test_existing_pair_on_secondary_cavity_component_is_preserved(self):
        cavity = self.mask | ((self.rows >= 185) & (self.rows <= 195) & (abs(self.cols-125) <= 7))
        regions = self.regions([[55, 155], [80, 155], [190, 80], [190, 155]])
        old = opposite_side_pairs(regions, cavity)
        self.assertEqual({*old['pairs'][0]['source_region_ids']}, {3, 4})
        recovered = complete_disjoint_pairs(regions, cavity, old)
        self.assertEqual([p['region_ids'] for p in recovered['pairs']], [[3, 4], [1, 2]])
        self.assertTrue(recovered['pairs'][0]['cavity_intersection_observed'])

    def test_degenerate_completion_does_not_draw_touching_lines(self):
        regions = self.regions([[100, 80], [100, 80], [100, 160], [100, 160]])
        old = opposite_side_pairs(regions, self.mask)
        before = [dict(r) for r in regions]
        self.assertIsNone(complete_disjoint_pairs(regions, self.mask, old))
        self.assertEqual(regions, before)

    def test_complete_pairings_are_untouched(self):
        regions = self.regions([[100, 80], [100, 160], [150, 80], [150, 160]])
        old = opposite_side_pairs(regions, self.mask)
        before = [dict(r) for r in regions]
        self.assertEqual(len(old['pairs']), 2)
        self.assertIsNone(complete_disjoint_pairs(regions, self.mask, old))
        self.assertEqual(regions, before)

    def test_every_detected_cusp_participates_exactly_once(self):
        cusps = np.zeros_like(self.mask)
        for r, c in [[55, 155], [80, 155], [190, 80], [190, 155]]:
            cusps |= ((self.rows-r)/10)**2+((self.cols-c)/8)**2 <= 1
        result = analyze_cusp_geometry(cusps, self.mask)
        self.assertTrue(result['disjoint_pair_completion_used'])
        self.assertEqual([r['region_id'] for r in result['regions']], [1, 2, 3, 4])
        self.assertEqual(sorted(i for p in result['pairs'] for i in p['region_ids']), [1, 2, 3, 4])
        validate_pair_set(result['pairs'])

    def test_completion_never_reintroduces_excluded_fifth_core_or_duplicate_ids(self):
        cusps = np.zeros_like(self.mask)
        for r, c in [[55, 155], [80, 155], [190, 80], [190, 155]]:
            cusps |= ((self.rows-r)/10)**2+((self.cols-c)/8)**2 <= 1
        cusps |= (self.rows-20)**2+(self.cols-70)**2 <= 4**2
        result = analyze_cusp_geometry(cusps, self.mask)
        self.assertTrue(result['disjoint_pair_completion_used'])
        self.assertEqual(len(result['regions']), 5)
        self.assertEqual(len({r['region_id'] for r in result['regions']}), 5)
        excluded = [r for r in result['regions'] if not r['pairing_eligible']]
        self.assertEqual([r['source_region_id'] for r in excluded], [1])
        self.assertEqual({i for p in result['pairs'] for i in p['source_region_ids']}, {2, 3, 4, 5})
        self.assertFalse(set(result['excluded_region_ids']) & {i for p in result['pairs'] for i in p['region_ids']})
        validate_pair_set(result['pairs'])



if __name__ == '__main__':
    unittest.main()
