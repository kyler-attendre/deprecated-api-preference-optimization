import unittest

from scripts.plot_library_radar import (
    LIBRARY_ORDER,
    build_radar_series,
    summarize_library_metrics,
)


class PlotLibraryRadarTest(unittest.TestCase):
    def test_summarize_library_metrics_computes_per_library_rates(self):
        rows = [
            {"library": "numpy", "has_deprecated": True, "has_replacement": False},
            {"library": "numpy", "has_deprecated": False, "has_replacement": True},
            {"library": "pytorch", "has_deprecated": False, "has_replacement": True},
            {"library": "pytorch", "has_deprecated": False, "has_replacement": False},
        ]

        metrics = summarize_library_metrics(rows, libraries=["numpy", "pytorch"])

        self.assertEqual(metrics["numpy"]["samples"], 2)
        self.assertAlmostEqual(metrics["numpy"]["deprecated_usage_rate"], 0.5)
        self.assertAlmostEqual(metrics["numpy"]["replacement_hit_rate"], 0.5)
        self.assertEqual(metrics["pytorch"]["samples"], 2)
        self.assertAlmostEqual(metrics["pytorch"]["deprecated_usage_rate"], 0.0)
        self.assertAlmostEqual(metrics["pytorch"]["replacement_hit_rate"], 0.5)

    def test_build_radar_series_follows_library_order_and_closes_loop(self):
        metrics = {
            "numpy": {"deprecated_usage_rate": 0.2, "replacement_hit_rate": 0.6},
            "pandas": {"deprecated_usage_rate": 0.1, "replacement_hit_rate": 0.4},
            "pytorch": {"deprecated_usage_rate": 0.3, "replacement_hit_rate": 0.8},
            "scipy": {"deprecated_usage_rate": 0.0, "replacement_hit_rate": 0.5},
            "seaborn": {"deprecated_usage_rate": 0.4, "replacement_hit_rate": 0.3},
            "sklearn": {"deprecated_usage_rate": 0.5, "replacement_hit_rate": 0.2},
            "tensorflow": {"deprecated_usage_rate": 0.6, "replacement_hit_rate": 0.1},
        }

        deprecated_series = build_radar_series(metrics, "deprecated_usage_rate")
        replacement_series = build_radar_series(metrics, "replacement_hit_rate")

        self.assertEqual(len(deprecated_series), len(LIBRARY_ORDER) + 1)
        self.assertEqual(len(replacement_series), len(LIBRARY_ORDER) + 1)
        self.assertAlmostEqual(deprecated_series[0], 0.2)
        self.assertAlmostEqual(deprecated_series[-1], deprecated_series[0])
        self.assertAlmostEqual(replacement_series[2], 0.8)
        self.assertAlmostEqual(replacement_series[-1], replacement_series[0])


if __name__ == "__main__":
    unittest.main()
