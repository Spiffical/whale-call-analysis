import unittest

from src.analysis.attention_localization import (
    AnnotationBox,
    box_iou,
    derive_annotation_frequency_bounds,
    interval_iou,
    mask_overlap_metrics,
    pointing_hit,
)


class AttentionLocalizationTests(unittest.TestCase):
    def test_box_iou_matches_expected_overlap(self) -> None:
        score = box_iou((0, 10, 0, 10), (5, 15, 5, 15))
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score, 25 / 175, places=6)

    def test_interval_iou_handles_disjoint_and_overlap(self) -> None:
        self.assertEqual(interval_iou(0, 10, 10, 20), 0.0)
        self.assertAlmostEqual(interval_iou(0, 10, 5, 15), 5 / 15, places=6)

    def test_mask_overlap_metrics_report_full_match(self) -> None:
        mask = [[False, False, False], [False, True, True], [False, True, True]]
        coverage, precision, iou = mask_overlap_metrics(mask=mask, gt_box=(1, 3, 1, 3))
        self.assertEqual(coverage, 1.0)
        self.assertEqual(precision, 1.0)
        self.assertEqual(iou, 1.0)

    def test_pointing_hit_checks_peak_inside_box(self) -> None:
        self.assertEqual(pointing_hit((5, 5), (4, 8, 4, 8)), 1.0)
        self.assertEqual(pointing_hit((3, 5), (4, 8, 4, 8)), 0.0)

    def test_frequency_bounds_fall_back_to_bucket_prior(self) -> None:
        annotation = AnnotationBox(
            annotation_id="ann_000001",
            filename="example.wav",
            species="Bp",
            call_type_bucket="20Hz",
            call_type_raw="20Hz",
            begin_time_s=0.0,
            end_time_s=10.0,
            low_freq_hz=None,
            high_freq_hz=None,
            peak_freq_hz=None,
            context_tags=("faint",),
            comments="",
        )
        low_hz, high_hz, source = derive_annotation_frequency_bounds(annotation)
        self.assertEqual((low_hz, high_hz, source), (12.0, 30.0, "bucket_prior"))


if __name__ == "__main__":
    unittest.main()

