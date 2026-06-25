import unittest

from scripts.plot_library_radar_panel import collect_panel_series


class PlotLibraryRadarPanelTest(unittest.TestCase):
    def test_collect_panel_series_extracts_one_metric_per_label_and_model(self):
        payloads = {
            "3b": {
                "metrics": {
                    "base": {"numpy": {"deprecated_usage_rate": 0.0}, "pytorch": {"deprecated_usage_rate": 0.2}},
                    "dpo": {"numpy": {"deprecated_usage_rate": 0.0}, "pytorch": {"deprecated_usage_rate": 0.0}},
                    "anchored_dpo": {"numpy": {"deprecated_usage_rate": 0.0}, "pytorch": {"deprecated_usage_rate": 0.0}},
                }
            }
        }

        panel = collect_panel_series(payloads, metric_name="deprecated_usage_rate", libraries=["numpy", "pytorch"])

        self.assertEqual(panel["3b"]["base"], [0.0, 0.2])
        self.assertEqual(panel["3b"]["dpo"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
