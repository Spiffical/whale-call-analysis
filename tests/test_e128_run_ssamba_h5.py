import unittest

from scripts.analysis import e128_run_ssamba_h5 as runner


class TestE128RunSSAMBAH5(unittest.TestCase):
    def parse(self, *extra):
        args = runner.build_parser().parse_args(
            [
                "--data-train",
                "/tmp/data.h5",
                "--exp-dir",
                "/tmp/run",
                *extra,
            ]
        )
        return runner.normalize_args(args)

    def test_parse_bool(self):
        self.assertTrue(runner.parse_bool("true"))
        self.assertTrue(runner.parse_bool("1"))
        self.assertFalse(runner.parse_bool("false"))
        self.assertFalse(runner.parse_bool("0"))

    def test_binary_ft_cls_uses_single_output_head(self):
        args = self.parse("--task", "ft_cls")
        self.assertFalse(args.multiclass)
        self.assertEqual(args.n_class, 2)
        self.assertEqual(args.num_classes, 1)
        self.assertEqual(args.main_metric, "auc")

    def test_multiclass_preserves_requested_classes(self):
        args = self.parse("--task", "ft_avgtok", "--multiclass", "--num_classes", "4")
        self.assertTrue(args.multiclass)
        self.assertEqual(args.n_class, 4)
        self.assertEqual(args.num_classes, 4)

    def test_pretrain_uses_ssl_binary_shape(self):
        args = self.parse("--task", "pretrain_joint")
        self.assertEqual(args.n_class, 2)
        self.assertEqual(args.num_classes, 2)
        self.assertEqual(args.main_metric, "acc")


if __name__ == "__main__":
    unittest.main()
