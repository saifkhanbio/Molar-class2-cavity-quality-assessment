"""Run: python3 -m unittest discover -s tests -p test_final_CQS_overall.py."""
import copy
import csv
import math
from pathlib import Path
import tempfile
import unittest

import final_CQS_overall as cqs


def source_tables():
    return {
        "occlusal_efd": {1: {"sample": "O_1_mask.png", "status": "OK", "cavity_type": "occlusal",
                              "average_matching_score_pct": "60", "similarity_pct": "99"}},
        "proximal_efd": {1: {"sample": "P_1_mask.png", "status": "OK", "cavity_type": "proximal",
                              "average_matching_score_pct": "80", "similarity_pct": "0"}},
        "depth": {1: {"Tooth": "O_1", "OcclusalDepthMedian_mm": "1.75", "GingivalFromPulpalMedian_mm": ".75",
                       "OcclusalDepthOffset_mm": ".75", "ProximalDepthMedian_mm": "99",
                       "OcclusalFloorResidualRMS_mm": "0", "ProximalFloorResidualRMS_mm": "0",
                       "OcclusalStatus": "measured_requires_review", "ProximalStatus": "measured_requires_review",
                       "GingivalFromPulpalStatus": "measured_requires_anatomical_review",
                       "GingivalFromPulpalWholeFloorMedian_mm": "1.25",
                       "Occlusal_bed_smooth": "20", "Proximal_bed_smooth": "20", "OcclusalMaxDepth(mm)": "99"}},
        "ratio": {1: {"filename": "O_1_mask.png", "status": "ratio_available",
                       "ratio_isthmus_to_intercuspal": str(1/3), "isthmus_width_px": "10", "intercuspal_distance_px": "30",
                       "within_near_tolerance": "True", "opposite_sides_valid": "True",
                       "crossing_checks_relaxed": "False", "shared_cusp_pair": "False"}},
    }


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(cqs.DEFAULT_CONFIG)
        self.tables = source_tables()

    def test_equal_shape_weights_and_unit_total(self):
        self.assertEqual(cqs.COMPONENTS["efd_occlusal"][1], .25)
        self.assertEqual(cqs.COMPONENTS["efd_proximal"][1], .25)
        self.assertAlmostEqual(sum(x[1] for x in cqs.COMPONENTS.values()), 1.)

    def test_perfect_components_sum_to_ten(self):
        total, points, weight, missing = cqs.aggregate({k: 1. for k in cqs.COMPONENTS})
        self.assertAlmostEqual(total, 10.)
        self.assertAlmostEqual(weight, 1.)
        self.assertEqual(missing, [])

    def test_uses_average_efd_latest_medians_and_rms_without_double_offset(self):
        row, detail = cqs.make_row(1, self.tables, self.config)
        self.assertAlmostEqual(row["CQS_overall"], 8.5)
        self.assertEqual(row["q_efd_occlusal"], .6)
        self.assertEqual(row["q_efd_proximal"], .8)
        self.assertEqual(row["q_depth_occlusal"], 1.)
        self.assertEqual(row["q_depth_gingival"], 1.)
        self.assertEqual(row["q_regularity_occlusal"], 1.)
        self.assertEqual(len(detail), 7)

    def test_occlusal_proximal_efd_swap_has_no_effect(self):
        before = cqs.make_row(1, self.tables, self.config)[0]["CQS_overall"]
        self.tables["occlusal_efd"][1]["average_matching_score_pct"] = "80"
        self.tables["proximal_efd"][1]["average_matching_score_pct"] = "60"
        self.assertAlmostEqual(cqs.make_row(1, self.tables, self.config)[0]["CQS_overall"], before)

    def test_depth_band_edges_and_taper(self):
        cfg = self.config["gingival_depth"]
        for raw, expected in [(-.68, 0.), (0., 0.), (.25, .5), (.5, 1.), (.75, 1.), (1., 1.), (1.25, .5), (1.5, 0.), (9., 0.)]:
            self.assertAlmostEqual(cqs.depth_score(raw, cfg), expected)

    def test_negative_raw_gingival_depth_is_preserved_and_scored_provisionally(self):
        self.tables["depth"][1]["GingivalFromPulpalMedian_mm"] = "-.678"
        row, _ = cqs.make_row(1, self.tables, self.config)
        self.assertEqual(row["GingivalFromPulpalMedian_mm"], -.678)
        self.assertEqual(row["q_depth_gingival"], 0.)
        self.assertAlmostEqual(row["CQS_overall"], 7.5)
        self.assertIn("negative_gingival_median", row["review_flags"])

    def test_missing_component_does_not_renormalize(self):
        scores = {k: 1. for k in cqs.COMPONENTS}
        scores["depth_gingival"] = None
        total, points, weight, missing = cqs.aggregate(scores)
        self.assertIsNone(total)
        self.assertAlmostEqual(weight, .9)
        self.assertAlmostEqual(sum(v for v in points.values() if v is not None), 9.)

    def test_missing_input_row_keeps_case_incomplete(self):
        self.tables["proximal_efd"] = {}
        row, _ = cqs.make_row(1, self.tables, self.config)
        self.assertIsNone(row["CQS_overall"])
        self.assertIn("efd_proximal", row["missing_components"])
        self.assertAlmostEqual(row["available_weight"], .75)

    def test_soft_ratio_target_half_height_and_log_symmetry(self):
        cfg = self.config["ratio"]
        self.assertAlmostEqual(cqs.ratio_score(1/3, cfg), 1.)
        self.assertAlmostEqual(cqs.ratio_score(.5, cfg), .5)
        self.assertAlmostEqual(cqs.ratio_score(2/9, cfg), .5)
        self.assertGreater(cqs.ratio_score(.2, cfg), 0.)
        self.assertAlmostEqual(cqs.ratio_score((1/3)*2.1, cfg), cqs.ratio_score((1/3)/2.1, cfg))

    def test_ratio_mismatch_is_not_silently_accepted(self):
        self.tables["ratio"][1]["ratio_isthmus_to_intercuspal"] = ".4"
        with self.assertRaisesRegex(ValueError, "width / distance"):
            cqs.make_row(1, self.tables, self.config)

    def test_source_review_flags_and_accepted_proximity_fallback_add_no_penalty(self):
        before = cqs.make_row(1, self.tables, self.config)[0]["CQS_overall"]
        self.tables["ratio"][1].update(status="ratio_requires_review", is_fallback="True", ratio_is_fallback="True", within_near_tolerance="False",
                                       ratio_fallback_stage="proximity_relaxed", warnings="review")
        self.tables["depth"][1]["GingivalFromPulpalGeometryFlags"] = "reference_extrapolation"
        self.tables["occlusal_efd"][1].update(status="REVIEW", warnings="disconnected_component")
        row, _ = cqs.make_row(1, self.tables, self.config)
        self.assertEqual(row["CQS_overall"], before)
        self.assertIn("proximity", row["review_flags"])

    def test_failed_source_status_cannot_reuse_stale_numeric_value(self):
        self.tables["depth"][1]["GingivalFromPulpalStatus"] = "unresolved"
        row, _ = cqs.make_row(1, self.tables, self.config)
        self.assertIsNone(row["CQS_overall"])
        self.assertIsNone(row["q_depth_gingival"])

    def test_hard_pair_geometry_violations_make_ratio_unavailable(self):
        self.tables["ratio"][1]["shared_cusp_pair"] = "True"
        row, _ = cqs.make_row(1, self.tables, self.config)
        self.assertIsNone(row["q_ratio"])
        self.assertIsNone(row["CQS_overall"])

    def test_reference_sensitivity_changes_only_gingival_contribution(self):
        row, _ = cqs.make_row(1, self.tables, self.config)
        self.assertAlmostEqual(row["CQS_if_whole_floor_gingival_reference"], 8.)
        self.assertAlmostEqual(row["CQS_reference_shift_points"], -.5)

    def test_invalid_values_are_missing_not_perfect_scores(self):
        for value in [None, "", "NaN", "inf", "abc"]:
            self.assertIsNone(cqs.number(value))
        self.assertIsNone(cqs.efd_score(101))
        self.assertIsNone(cqs.efd_score(-1))
        self.assertIsNone(cqs.regularity_score(-.1, .5))
        self.assertAlmostEqual(cqs.regularity_score(.5, .5), math.exp(-1))
        self.assertIsNone(cqs.ratio_score(0, self.config["ratio"]))

    def test_invalid_configuration_is_rejected(self):
        self.config["gingival_depth"]["tolerance_mm"] = 0
        with self.assertRaises(ValueError):
            cqs.validate_config(self.config)


class JoinTests(unittest.TestCase):
    def test_case_normalization_across_views_and_suffixes(self):
        for name in ["O_11_st_mask.png", "P_011_mask.png", "O-11.stl", "O_11"]:
            self.assertEqual(cqs.case_key(name), 11)
        with self.assertRaises(ValueError):
            cqs.case_key("unrelated11.csv")

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"scores.csv"
            path.write_text("sample,score\nO_1_mask.png,90\nO_01_st_mask.png,80\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                cqs.read_table(path, "sample", ["score"])

    def test_obsolete_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"scores.csv"
            path.write_text("sample,similarity_pct\nO_1_mask.png,99\n")
            with self.assertRaisesRegex(ValueError, "missing required"):
                cqs.read_table(path, "sample", ["average_matching_score_pct"])


class PublicationInputTests(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(cqs.DEFAULT_CONFIG)
        self.tables = source_tables()
        self.depth = self.tables["depth"][1]
        self.depth.update(ProximalDepthMedian_mm="2.5", GingivalMinusPulpalDepth_mm=".75",
                          GingivalMinusPulpalMethod="difference_of_area_weighted_medians_same_crown_reference",
                          Error="")

    def test_publication_measurement_overrides_stale_local_plane_fields(self):
        self.depth.update(GingivalFromPulpalMedian_mm="9", GingivalFromPulpalStatus="unresolved",
                          GingivalFromPulpalGeometryFlags="stale_local_plane_flag")
        row, _ = cqs.make_row(1, self.tables, self.config)
        self.assertEqual(row["GingivalFromPulpalMedian_mm"], .75)
        self.assertEqual(row["GingivalMinusPulpalDepth_mm"], .75)
        self.assertEqual(row["ProximalDepthMedian_mm"], 2.5)
        self.assertAlmostEqual(row["CQS_overall"], 8.5)
        self.assertIsNone(row["CQS_if_whole_floor_gingival_reference"])
        self.assertNotIn("stale_local_plane_flag", row["review_flags"])
        self.assertNotIn("OcclusalDepthOffset_mm_already_applied", row)

    def test_difference_must_equal_proximal_minus_occlusal(self):
        self.depth["GingivalMinusPulpalDepth_mm"] = "1.5"
        with self.assertRaisesRegex(ValueError, "proximal minus occlusal"):
            cqs.make_row(1, self.tables, self.config)

    def test_either_failed_floor_invalidates_relative_depth(self):
        for column in ("OcclusalStatus", "ProximalStatus"):
            with self.subTest(column=column):
                tables = copy.deepcopy(self.tables)
                tables["depth"][1][column] = "not_detected"
                row, _ = cqs.make_row(1, tables, self.config)
                self.assertIsNone(row["q_depth_gingival"])
                self.assertIsNone(row["CQS_overall"])
                self.assertIn("depth_gingival", row["missing_components"])

    def test_missing_floor_median_cannot_reuse_old_local_depth(self):
        self.depth["ProximalDepthMedian_mm"] = ""
        row, _ = cqs.make_row(1, self.tables, self.config)
        self.assertIsNone(row["GingivalFromPulpalMedian_mm"])
        self.assertIsNone(row["q_depth_gingival"])

    def test_negative_publication_difference_is_retained(self):
        self.depth.update(ProximalDepthMedian_mm="1.35", GingivalMinusPulpalDepth_mm="-.4")
        row, _ = cqs.make_row(1, self.tables, self.config)
        self.assertEqual(row["GingivalFromPulpalMedian_mm"], -.4)
        self.assertEqual(row["q_depth_gingival"], 0.)
        self.assertAlmostEqual(row["CQS_overall"], 7.5)

    def test_publication_schema_reads_without_legacy_status_columns(self):
        row = {key: value for key, value in self.depth.items() if not key.startswith("GingivalFromPulpal")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"depth_summary.csv"
            cqs.write_csv(path, [row])
            table, schema = cqs.read_depth_table(path)
        self.assertEqual(schema, "publication")
        self.assertEqual(table[1]["GingivalMinusPulpalDepth_mm"], ".75")

    def test_minimal_export_preserves_chosen_columns_and_full_precision(self):
        row, _ = cqs.make_row(1, self.tables, self.config)
        columns = ["case_id", "Tooth", "CQS_overall", "GingivalFromPulpalMedian_mm"]
        with tempfile.TemporaryDirectory() as directory:
            schema, output = Path(directory)/"schema.csv", Path(directory)/"minimal.csv"
            schema.write_text(",".join(columns)+"\n")
            self.assertEqual(cqs.write_minimal_csv(output, [row], schema), columns)
            with output.open() as stream:
                exported = next(csv.DictReader(stream))
            self.assertEqual(list(exported), columns)
            self.assertEqual(float(exported["CQS_overall"]), row["CQS_overall"])
            self.assertEqual(float(exported["GingivalFromPulpalMedian_mm"]), .75)


if __name__ == "__main__":
    unittest.main()
