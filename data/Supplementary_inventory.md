# Supplementary material inventory

Designated repository: https://github.com/saifkhanbio/Molar-class2-cavity-quality-assessment

S1 — CQS_overall_results: all case scores, components, expert ratings, and score displays.
S2 — final_avg_efd_results: occlusal/proximal scores, unique references, pairwise comparisons, and descriptor details. Figure Collection S1 contains all contour panels.
S3 — cus_ist_ratio_results: widths, cusp centers, distances, ratios, candidate pairings, detailed records. Figure Collection S2 contains all cusp/isthmus displays.
S4 — publication_cavity_depth_results: depth_summary.csv, render_manifest.json, all depth PNGs, previews, and interactive HTML. Figure Collection S3 contains the full 21-case depth series.
S5 — scripts/model_performance: model metrics, training logs, histories, checkpoint metadata, and retained-mask correspondence.
Supplementary Methods S1 — scripts: notebook, analysis and landmark code, model checkpoints, environment requirements, source inventory, and tests. The pred_* folders contain retained masks.
S6 — the three CSV files in this folder: manuscript-level numerical summaries, additional agreement calculations and bootstrap intervals, and casewise expert comparisons.

The new S6 files were calculated from the retained results; no model was retrained and no source measurements were changed. Bootstrap resampling used 10,000 paired case resamples and seed 20260923. Cases 13 and 18 appear in the main illustrations; all remaining case displays stay in their existing result folders. This local package does not upload files to GitHub.
