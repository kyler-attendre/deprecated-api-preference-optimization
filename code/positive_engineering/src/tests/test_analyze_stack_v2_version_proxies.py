import unittest

from scripts.analyze_stack_v2_version_proxies import (
    classify_version_proxy_path,
    summarize_version_proxy_rows,
)


class AnalyzeStackV2VersionProxiesTest(unittest.TestCase):
    def test_classify_version_proxy_path_tags_expected_carriers(self):
        self.assertEqual(
            classify_version_proxy_path("/project/setup.py"),
            {"setup_py"},
        )
        self.assertEqual(
            classify_version_proxy_path("/pkg/_version.py"),
            {"version_module_py"},
        )
        self.assertEqual(
            classify_version_proxy_path("/docs/conf.py"),
            {"docs_conf_py"},
        )
        self.assertEqual(
            classify_version_proxy_path("/src/app/settings_prod.py"),
            {"config_or_settings_py"},
        )
        self.assertEqual(
            classify_version_proxy_path("/lib/__init__.py"),
            {"init_py"},
        )
        self.assertEqual(
            classify_version_proxy_path("/tests/test_version.py"),
            {"examples_or_tests_path", "version_module_py"},
        )

    def test_summarize_version_proxy_rows_counts_categories_and_examples(self):
        rows = [
            {"path": "/project/setup.py", "repo_name": "repo/a"},
            {"path": "/pkg/_version.py", "repo_name": "repo/a"},
            {"path": "/docs/conf.py", "repo_name": "repo/b"},
            {"path": "/src/app/settings_prod.py", "repo_name": "repo/c"},
            {"path": "/tests/test_version.py", "repo_name": "repo/d"},
        ]

        summary = summarize_version_proxy_rows(rows)

        self.assertEqual(summary["rows_scanned"], 5)
        self.assertEqual(summary["category_counts"]["setup_py"], 1)
        self.assertEqual(summary["category_counts"]["version_module_py"], 2)
        self.assertEqual(summary["category_counts"]["docs_conf_py"], 1)
        self.assertEqual(summary["category_counts"]["config_or_settings_py"], 1)
        self.assertEqual(summary["category_counts"]["examples_or_tests_path"], 1)
        self.assertAlmostEqual(summary["category_rates"]["version_module_py"], 0.4)
        self.assertEqual(summary["example_paths"]["setup_py"][0], "/project/setup.py")


if __name__ == "__main__":
    unittest.main()
