"""
Tests for bigquery_pipeline/clean.py.

Two layers:
  - Mock-based: verify SQL generation and Python logic (no BQ, no network)
  - DuckDB integration: execute the real CTE chain against in-memory rows,
    assert all 3 duplicate types are resolved correctly

Run with: pytest tests/dataset_tools/test_clean.py -v
"""
import json
import types
from argparse import Namespace
from unittest.mock import MagicMock, call, patch

import duckdb
import pandas as pd
import pytest

import pytest

import src.dataset_tools.bigquery_pipeline.clean as clean


# ── helpers ───────────────────────────────────────────────────────────────────

def make_count_row(stage, n, taxa=100):
    """Minimal BQ row mock matching the count query SELECT (stage, n, taxa)."""
    row = MagicMock()
    row.stage = stage
    row.n = n
    row.taxa = taxa
    return row


SAMPLE_COUNTS = {
    "input":     {"rows": 37_500, "taxa": 5_000},
    "step1":     {"rows": 27_000, "taxa": 5_000},
    "step2":     {"rows": 25_700, "taxa": 5_000},
    "step3":     {"rows": 21_000, "taxa": 4_700},
    "step4":     {"rows": 21_000, "taxa": 4_700},
    "conflicts": {"rows": 2_229,  "taxa": 0},
}


def make_bq_client(counts=SAMPLE_COUNTS):
    """Mock BQ client whose query().result() returns the given stage counts."""
    rows = [make_count_row(stage, v["rows"], v["taxa"]) for stage, v in counts.items()]
    client = MagicMock()
    client.query.return_value.result.return_value = rows
    return client


def make_args(**kwargs):
    """Build a minimal Namespace matching clean.py's argparse output."""
    defaults = dict(
        dataset="global_all_leps_2605",
        project="leps-ai",
        table="training_images",
        min_images_per_taxon=0,
        multi_taxon_strategy="drop",
        dry_run=False,
        log_file=None,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


# ── _step3_cte ────────────────────────────────────────────────────────────────

class TestStep3Cte:

    def test_drop_strategy_uses_not_in(self):
        sql = clean._step3_cte("drop")
        assert "NOT IN" in sql
        assert "multi_taxon_photos" in sql

    def test_keep_lowest_uses_qualify_row_number(self):
        sql = clean._step3_cte("keep-lowest-taxon-id")
        assert "QUALIFY" in sql
        assert "ROW_NUMBER()" in sql
        assert "PARTITION BY photo_id" in sql
        assert "ORDER BY inat_taxon_id" in sql

    def test_drop_does_not_contain_qualify(self):
        sql = clean._step3_cte("drop")
        assert "QUALIFY" not in sql

    def test_keep_lowest_does_not_contain_not_in(self):
        sql = clean._step3_cte("keep-lowest-taxon-id")
        assert "NOT IN" not in sql


# ── _cte_chain SQL content ────────────────────────────────────────────────────

class TestCteSqlContent:

    SRC = "leps-ai.global_all_leps_2605.training_images"

    def _chain(self, strategy="drop", min_images=0):
        return clean._cte_chain(self.SRC, strategy, min_images)

    def test_src_ref_embedded(self):
        assert self.SRC in self._chain()

    def test_step1_exact_dedup_partition(self):
        sql = self._chain()
        assert "PARTITION BY photo_id, inat_taxon_id, gbif_id" in sql

    def test_step1_uses_qualify_row_number(self):
        sql = self._chain()
        assert "QUALIFY" in sql
        assert "ROW_NUMBER()" in sql

    def test_step2_same_taxon_multi_gbif_partition(self):
        sql = self._chain()
        assert "PARTITION BY photo_id, inat_taxon_id" in sql

    def test_step2_orders_by_gbif_id(self):
        sql = self._chain()
        assert "ORDER BY gbif_id" in sql

    def test_multi_taxon_photos_cte_present(self):
        sql = self._chain()
        assert "multi_taxon_photos" in sql
        assert "COUNT(DISTINCT inat_taxon_id) > 1" in sql

    def test_step4_min_images_filter_active(self):
        sql = self._chain(min_images=10)
        assert "t.cnt >= 10" in sql

    def test_step4_min_images_disabled_when_zero(self):
        sql = self._chain(min_images=0)
        assert "TRUE" in sql
        assert "t.cnt >=" not in sql

    def test_step3_drop_strategy_in_chain(self):
        sql = self._chain(strategy="drop")
        assert "NOT IN" in sql

    def test_step3_keep_lowest_strategy_in_chain(self):
        sql = self._chain(strategy="keep-lowest-taxon-id")
        assert "ORDER BY inat_taxon_id" in sql


# ── run_count_query ───────────────────────────────────────────────────────────

class TestRunCountQuery:

    SRC = "leps-ai.global_all_leps_2605.training_images_test"

    def test_returns_dict_with_all_stages(self):
        client = make_bq_client()
        result = clean.run_count_query(client, self.SRC, "drop", 0)
        assert set(result.keys()) == {"input", "step1", "step2", "step3", "step4", "conflicts"}

    def test_rows_parsed_correctly(self):
        client = make_bq_client()
        result = clean.run_count_query(client, self.SRC, "drop", 0)
        assert result["input"]["rows"] == 37_500
        assert result["step4"]["rows"] == 21_000
        assert result["conflicts"]["rows"] == 2_229

    def test_taxa_parsed_correctly(self):
        client = make_bq_client()
        result = clean.run_count_query(client, self.SRC, "drop", 0)
        assert result["input"]["taxa"] == 5_000
        assert result["step3"]["taxa"] == 4_700

    def test_bq_client_query_called_once(self):
        client = make_bq_client()
        clean.run_count_query(client, self.SRC, "drop", 0)
        assert client.query.call_count == 1

    def test_result_called_on_query(self):
        client = make_bq_client()
        clean.run_count_query(client, self.SRC, "drop", 0)
        client.query.return_value.result.assert_called_once()

    def test_sql_contains_union_all_for_all_stages(self):
        client = make_bq_client()
        clean.run_count_query(client, self.SRC, "drop", 0)
        sql = client.query.call_args[0][0]
        assert sql.count("UNION ALL") >= 5

    def test_sql_contains_src_ref(self):
        client = make_bq_client()
        clean.run_count_query(client, self.SRC, "drop", 0)
        sql = client.query.call_args[0][0]
        assert self.SRC in sql

    def test_drop_strategy_sql_has_not_in(self):
        client = make_bq_client()
        clean.run_count_query(client, self.SRC, "drop", 0)
        sql = client.query.call_args[0][0]
        assert "NOT IN" in sql

    def test_keep_lowest_strategy_sql_has_qualify(self):
        client = make_bq_client()
        clean.run_count_query(client, self.SRC, "keep-lowest-taxon-id", 0)
        sql = client.query.call_args[0][0]
        assert "ORDER BY inat_taxon_id" in sql


# ── run_write_query ───────────────────────────────────────────────────────────

class TestRunWriteQuery:

    SRC = "leps-ai.global_all_leps_2605.training_images"
    DST = "leps-ai.global_all_leps_2605.training_images"

    def test_issues_create_or_replace_table(self):
        client = MagicMock()
        clean.run_write_query(client, self.SRC, self.DST, "drop", 0)
        sql = client.query.call_args[0][0]
        assert "CREATE OR REPLACE TABLE" in sql

    def test_dst_ref_in_sql(self):
        client = MagicMock()
        clean.run_write_query(client, self.SRC, self.DST, "drop", 0)
        sql = client.query.call_args[0][0]
        assert self.DST in sql

    def test_selects_from_step4(self):
        client = MagicMock()
        clean.run_write_query(client, self.SRC, self.DST, "drop", 0)
        sql = client.query.call_args[0][0]
        assert "SELECT * FROM step4" in sql

    def test_result_called(self):
        client = MagicMock()
        clean.run_write_query(client, self.SRC, self.DST, "drop", 0)
        client.query.return_value.result.assert_called_once()

    def test_query_called_exactly_once(self):
        client = MagicMock()
        clean.run_write_query(client, self.SRC, self.DST, "drop", 0)
        assert client.query.call_count == 1


# ── build_log ─────────────────────────────────────────────────────────────────

class TestBuildLog:

    STARTED = "2026-06-04T01:00:00+00:00"

    def _log(self, counts=SAMPLE_COUNTS, **kwargs):
        args = make_args(**kwargs)
        return clean.build_log(counts, args, "leps-ai.ds.training_images", args.dry_run, self.STARTED)

    def test_summary_input_rows(self):
        assert self._log()["summary"]["input_rows"] == 37_500

    def test_summary_output_rows(self):
        assert self._log()["summary"]["output_rows"] == 21_000

    def test_summary_total_removed(self):
        log = self._log()
        assert log["summary"]["total_removed"] == 37_500 - 21_000

    def test_summary_taxa_before_and_after(self):
        log = self._log()
        assert log["summary"]["taxa_before"] == 5_000
        assert log["summary"]["taxa_after"] == 4_700

    def test_step1_removed_arithmetic(self):
        log = self._log()
        step = log["steps"]["exact_duplicates"]
        assert step["removed"] == 37_500 - 27_000

    def test_step2_removed_arithmetic(self):
        log = self._log()
        step = log["steps"]["same_taxon_multi_gbif"]
        assert step["removed"] == 27_000 - 25_700

    def test_step3_removed_arithmetic(self):
        log = self._log()
        step = log["steps"]["multi_taxon_conflicts"]
        assert step["removed"] == 25_700 - 21_000

    def test_step3_conflict_photo_ids(self):
        log = self._log()
        assert log["steps"]["multi_taxon_conflicts"]["conflict_photo_ids"] == 2_229

    def test_step4_threshold_recorded(self):
        log = self._log(min_images_per_taxon=10)
        assert log["steps"]["min_images_filter"]["threshold"] == 10

    def test_dry_run_flag_true(self):
        assert self._log(dry_run=True)["dry_run"] is True

    def test_dry_run_flag_false(self):
        assert self._log(dry_run=False)["dry_run"] is False

    def test_started_at_preserved(self):
        assert self._log()["started_at"] == self.STARTED

    def test_log_is_json_serialisable(self):
        log = self._log()
        json.dumps(log)  # must not raise


# ── print_report ──────────────────────────────────────────────────────────────

class TestPrintReport:

    def test_no_crash(self, capsys):
        clean.print_report(SAMPLE_COUNTS, "drop", 0, "leps-ai.ds.t", dry_run=False)
        capsys.readouterr()  # consume output

    def test_dry_run_tag_present(self, capsys):
        clean.print_report(SAMPLE_COUNTS, "drop", 0, "leps-ai.ds.t", dry_run=True)
        out = capsys.readouterr().out
        assert "DRY RUN" in out

    def test_dry_run_tag_absent_when_not_dry(self, capsys):
        clean.print_report(SAMPLE_COUNTS, "drop", 0, "leps-ai.ds.t", dry_run=False)
        out = capsys.readouterr().out
        assert "DRY RUN" not in out

    def test_total_removed_in_output(self, capsys):
        clean.print_report(SAMPLE_COUNTS, "drop", 0, "leps-ai.ds.t", dry_run=False)
        out = capsys.readouterr().out
        assert "16,500" in out  # 37500 - 21000

    def test_conflict_photo_ids_in_output(self, capsys):
        clean.print_report(SAMPLE_COUNTS, "drop", 0, "leps-ai.ds.t", dry_run=False)
        out = capsys.readouterr().out
        assert "2,229" in out

    def test_strategy_name_in_output(self, capsys):
        clean.print_report(SAMPLE_COUNTS, "keep-lowest-taxon-id", 0, "leps-ai.ds.t", dry_run=False)
        out = capsys.readouterr().out
        assert "keep-lowest-taxon-id" in out


# ── main() integration ────────────────────────────────────────────────────────

class TestMainIntegration:

    def _run_main(self, argv, counts=SAMPLE_COUNTS):
        client = make_bq_client(counts)
        with patch("sys.argv", ["clean.py"] + argv), \
             patch("src.dataset_tools.bigquery_pipeline.clean.bigquery.Client",
                   return_value=client):
            clean.main()
        return client

    def test_dry_run_calls_count_query_not_write(self):
        client = self._run_main([
            "--dataset", "global_all_leps_2605",
            "--table", "training_images_test",
            "--dry-run",
        ])
        assert client.query.call_count == 1
        sql = client.query.call_args[0][0]
        assert "CREATE OR REPLACE TABLE" not in sql

    def test_actual_run_calls_count_then_write(self):
        client = self._run_main([
            "--dataset", "global_all_leps_2605",
            "--table", "training_images_test",
        ])
        assert client.query.call_count == 2
        write_sql = client.query.call_args_list[1][0][0]
        assert "CREATE OR REPLACE TABLE" in write_sql

    def test_log_file_written(self, tmp_path):
        log_path = tmp_path / "clean.json"
        self._run_main([
            "--dataset", "global_all_leps_2605",
            "--dry-run",
            "--log-file", str(log_path),
        ])
        assert log_path.exists()
        log = json.loads(log_path.read_text())
        assert "summary" in log
        assert "steps" in log

    def test_min_images_arg_propagated_to_sql(self):
        client = self._run_main([
            "--dataset", "global_all_leps_2605",
            "--min-images-per-taxon", "15",
            "--dry-run",
        ])
        sql = client.query.call_args[0][0]
        assert "t.cnt >= 15" in sql

    def test_keep_lowest_strategy_propagated_to_sql(self):
        client = self._run_main([
            "--dataset", "global_all_leps_2605",
            "--multi-taxon-strategy", "keep-lowest-taxon-id",
            "--dry-run",
        ])
        sql = client.query.call_args[0][0]
        assert "ORDER BY inat_taxon_id" in sql


# ── DuckDB integration — real SQL against in-memory rows ─────────────────────

@pytest.fixture
def dup_df():
    """
    Small DataFrame covering all 3 duplicate types plus clean rows.

    photo_id=1  type 1 — exact duplicate (identical rows)
    photo_id=2  type 2 — same photo, two different taxa + gbif_ids
    photo_id=3  type 3 — same photo + taxon, two different gbif_ids
    photo_id=4  clean
    photo_id=5  clean, different taxon
    photo_id=6  clean, used for min-images-per-taxon tests (sole photo of taxon 600)
    """
    return pd.DataFrame([
        # Type 1: exact dup
        dict(photo_id=1, inat_taxon_id=100, gbif_id=1000, dataset_source_uuid="uuid-1"),
        dict(photo_id=1, inat_taxon_id=100, gbif_id=1000, dataset_source_uuid="uuid-1"),
        # Type 2: multi-taxon conflict
        dict(photo_id=2, inat_taxon_id=200, gbif_id=2000, dataset_source_uuid="uuid-2"),
        dict(photo_id=2, inat_taxon_id=201, gbif_id=2001, dataset_source_uuid="uuid-2"),
        # Type 3: same taxon, multi-gbif (keep MIN gbif_id=3000)
        dict(photo_id=3, inat_taxon_id=300, gbif_id=3000, dataset_source_uuid="uuid-3"),
        dict(photo_id=3, inat_taxon_id=300, gbif_id=3001, dataset_source_uuid="uuid-3"),
        # Clean rows
        dict(photo_id=4, inat_taxon_id=100, gbif_id=4000, dataset_source_uuid="uuid-4"),
        dict(photo_id=5, inat_taxon_id=300, gbif_id=5000, dataset_source_uuid="uuid-5"),
        # Lone taxon (taxon 600 has only 1 image — for min-images test)
        dict(photo_id=6, inat_taxon_id=600, gbif_id=6000, dataset_source_uuid="uuid-6"),
    ])


def run_cte(df, strategy="drop", min_images=0):
    """Execute the clean.py CTE chain against a pandas DataFrame via DuckDB."""
    conn = duckdb.connect()
    conn.register("src_table", df)
    cte = clean._cte_chain("src_table", strategy, min_images).replace("`", "")
    return conn.execute(f"WITH {cte} SELECT * FROM step4").df()


class TestDuckDBIntegration:
    """Execute the real CTE SQL against in-memory rows — no BQ required."""

    def test_output_row_count(self, dup_df):
        result = run_cte(dup_df)
        # type1: 1 kept, type2: 0 kept (both dropped), type3: 1 kept, clean: 3 kept
        assert len(result) == 5

    def test_type1_exact_dup_resolved(self, dup_df):
        result = run_cte(dup_df)
        assert result[result.photo_id == 1].shape[0] == 1

    def test_type2_multi_taxon_dropped(self, dup_df):
        result = run_cte(dup_df)
        assert result[result.photo_id == 2].shape[0] == 0

    def test_type3_keeps_min_gbif_id(self, dup_df):
        result = run_cte(dup_df)
        row = result[result.photo_id == 3]
        assert len(row) == 1
        assert row.iloc[0].gbif_id == 3000

    def test_clean_rows_preserved(self, dup_df):
        result = run_cte(dup_df)
        assert result[result.photo_id == 4].shape[0] == 1
        assert result[result.photo_id == 5].shape[0] == 1

    def test_no_duplicate_photo_ids_remain(self, dup_df):
        result = run_cte(dup_df)
        assert result.photo_id.nunique() == len(result)

    def test_no_duplicate_uuids_remain(self, dup_df):
        result = run_cte(dup_df)
        assert result.dataset_source_uuid.nunique() == len(result)

    def test_each_photo_has_one_taxon(self, dup_df):
        result = run_cte(dup_df)
        taxa_per_photo = result.groupby("photo_id")["inat_taxon_id"].nunique()
        assert (taxa_per_photo == 1).all()

    def test_min_images_drops_lone_taxon(self, dup_df):
        # taxon 600 has only 1 image — threshold=2 should drop it
        result = run_cte(dup_df, min_images=2)
        assert result[result.inat_taxon_id == 600].shape[0] == 0

    def test_min_images_keeps_taxon_above_threshold(self, dup_df):
        # taxon 100 has 2 images (photo_id 1 and 4) — survives threshold=2
        result = run_cte(dup_df, min_images=2)
        assert result[result.inat_taxon_id == 100].shape[0] == 2

    def test_keep_lowest_taxon_keeps_one_row_per_photo(self, dup_df):
        result = run_cte(dup_df, strategy="keep-lowest-taxon-id")
        assert result[result.photo_id == 2].shape[0] == 1

    def test_keep_lowest_taxon_picks_min_taxon_id(self, dup_df):
        result = run_cte(dup_df, strategy="keep-lowest-taxon-id")
        row = result[result.photo_id == 2]
        assert row.iloc[0].inat_taxon_id == 200  # min(200, 201)

    # ── edge cases ────────────────────────────────────────────────────────────

    def test_type1_and_type3_overlap_resolved_to_one_row(self):
        """Photo with both exact-dup rows AND multiple gbif_ids under same taxon."""
        df = pd.DataFrame([
            dict(photo_id=1, inat_taxon_id=100, gbif_id=1000, dataset_source_uuid="uuid-1"),
            dict(photo_id=1, inat_taxon_id=100, gbif_id=1000, dataset_source_uuid="uuid-1"),
            dict(photo_id=1, inat_taxon_id=100, gbif_id=1001, dataset_source_uuid="uuid-1"),
        ])
        result = run_cte(df)
        assert len(result) == 1
        assert result.iloc[0].gbif_id == 1000

    def test_type1_with_three_copies_resolves_to_one(self):
        """Three identical rows (tripled duplicate) collapse to one."""
        df = pd.DataFrame([
            dict(photo_id=1, inat_taxon_id=100, gbif_id=1000, dataset_source_uuid="uuid-1"),
            dict(photo_id=1, inat_taxon_id=100, gbif_id=1000, dataset_source_uuid="uuid-1"),
            dict(photo_id=1, inat_taxon_id=100, gbif_id=1000, dataset_source_uuid="uuid-1"),
        ])
        result = run_cte(df)
        assert len(result) == 1

    def test_type3_with_three_gbif_ids_keeps_min(self):
        """Same photo+taxon with 3 different gbif_ids — keeps the lowest."""
        df = pd.DataFrame([
            dict(photo_id=1, inat_taxon_id=100, gbif_id=3000, dataset_source_uuid="uuid-1"),
            dict(photo_id=1, inat_taxon_id=100, gbif_id=3001, dataset_source_uuid="uuid-1"),
            dict(photo_id=1, inat_taxon_id=100, gbif_id=3002, dataset_source_uuid="uuid-1"),
        ])
        result = run_cte(df)
        assert len(result) == 1
        assert result.iloc[0].gbif_id == 3000

    def test_empty_table_returns_zero_rows(self):
        """Empty input table produces empty output without error."""
        df = pd.DataFrame(
            columns=["photo_id", "inat_taxon_id", "gbif_id", "dataset_source_uuid"]
        )
        result = run_cte(df)
        assert len(result) == 0
