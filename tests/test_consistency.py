"""Cross-file consistency checks.

These catch the class of bug that has no single owner: a column renamed in
schema.sql but not in sites.yaml, a compose variable that never made it into
.env.example, an eval question pointing at a site that does not exist. Each is
trivially fixable and completely silent until something fails at 3am.
"""

from __future__ import annotations

import re

import pytest
import yaml
from dotenv import dotenv_values

from metermind.config import PROJECT_ROOT, SITES_FILE
from metermind.data.load_postgres import SITE_COLUMNS

VALID_CATEGORIES = {"manufacturing", "office", "cold_storage", "retail"}
VALID_EVAL_CATEGORIES = {"timeseries_lookup", "anomaly", "contextual_rag"}


@pytest.fixture(scope="module")
def sites() -> list[dict]:
    return yaml.safe_load(SITES_FILE.read_text(encoding="utf-8"))["sites"]


@pytest.fixture(scope="module")
def eval_set() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "eval" / "questions.yaml").read_text(encoding="utf-8"))


class TestSiteRegister:
    def test_every_site_has_every_postgres_column(self, sites):
        for site in sites:
            missing = [column for column in SITE_COLUMNS if column not in site]
            assert not missing, f"{site.get('site_id')} is missing {missing}"

    def test_site_ids_are_unique(self, sites):
        ids = [site["site_id"] for site in sites]
        assert len(ids) == len(set(ids))

    def test_categories_are_known(self, sites):
        for site in sites:
            assert site["category"] in VALID_CATEGORIES

    def test_every_category_has_at_least_two_sites(self, sites):
        """Single-site categories make "which of our X sites" questions trivial."""
        counts = {category: 0 for category in VALID_CATEGORIES}
        for site in sites:
            counts[site["category"]] += 1
        assert all(count >= 2 for count in counts.values()), counts

    def test_base_load_below_peak_load(self, sites):
        for site in sites:
            assert 0 < site["base_load_kw"] < site["peak_load_kw"], site["site_id"]

    def test_cold_storage_has_the_highest_base_to_peak_ratio(self, sites):
        """Encodes the domain claim made in sites.yaml.

        A cold store runs near capacity around the clock; an office sits near
        base load most of the week. If a profile model is ever fitted to these
        ratios, this is the assumption it inherits.
        """
        ratio = {s["site_id"]: s["base_load_kw"] / s["peak_load_kw"] for s in sites}
        cold = [s["site_id"] for s in sites if s["category"] == "cold_storage"]
        office = [s["site_id"] for s in sites if s["category"] == "office"]
        assert min(ratio[c] for c in cold) > max(ratio[o] for o in office)


class TestComposeAndEnv:
    def test_all_compose_variables_are_defined(self):
        compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        declared = set(dotenv_values(PROJECT_ROOT / ".env.example"))
        # Matches ${VAR} and ${VAR:-default}. Without the optional default
        # group, a variable written with a fallback silently escapes this check.
        referenced = set(re.findall(r"\$\{(\w+)(?::-[^}]*)?\}", compose))
        assert not (referenced - declared), f"undefined in .env.example: {referenced - declared}"

    def test_compose_is_valid_yaml_with_both_services(self):
        compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        assert set(compose["services"]) == {"influxdb", "postgres"}

    def test_env_is_gitignored_but_example_is_not(self):
        ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert re.search(r"^\.env$", ignored, re.MULTILINE)

    def test_generated_data_is_not_gitignored(self):
        """data/generated holds committed ground truth the eval scores against."""
        ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert not re.search(r"^data/generated/?$", ignored, re.MULTILINE)


class TestEvalSet:
    def test_question_ids_are_unique(self, eval_set):
        ids = [question["id"] for question in eval_set["questions"]]
        assert len(ids) == len(set(ids))

    def test_categories_are_known(self, eval_set):
        for question in eval_set["questions"]:
            assert question["category"] in VALID_EVAL_CATEGORIES

    def test_every_referenced_site_exists(self, eval_set, sites):
        known = {site["site_id"] for site in sites}
        for question in eval_set["questions"]:
            args = question["verify"].get("args", {})
            if "site_id" in args:
                assert args["site_id"] in known, f"{question['id']}: unknown site"

    def test_every_question_has_a_verifier_and_a_rationale(self, eval_set):
        for question in eval_set["questions"]:
            assert question["verify"].get("fn"), f"{question['id']}: no verifier"
            assert question.get("why", "").strip(), f"{question['id']}: no rationale"

    def test_all_three_categories_are_represented(self, eval_set):
        present = {question["category"] for question in eval_set["questions"]}
        assert present == VALID_EVAL_CATEGORIES

    def test_unbound_questions_are_flagged(self, eval_set):
        """A placeholder must be declared, not left to be discovered by a
        confusing eval failure."""
        for question in eval_set["questions"]:
            args = str(question["verify"].get("args", {}))
            if "TODO" in args:
                assert question["status"] != "ready", f"{question['id']}: TODO but marked ready"
