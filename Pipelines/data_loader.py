"""
data_loader.py  —  ARYAN  (Task 1)
Reads Parquet files using DuckDB with filtering + batch loading.
Target: 1M rows in under 30 seconds.

HOW TO PLUG IN YOUR DATA:
  loader = DataLoader("path/to/real_data_combined.parquet")
  batches = loader.get_batches(batch_size=10000)
"""

import duckdb
import pandas as pd
import time
import os


class DataLoader:
    """
    High-speed Parquet loader using DuckDB.
    Supports filtering, batch loading, and schema inspection.
    """

    def __init__(self, parquet_path: str = None):
        self.con = duckdb.connect()
        self.parquet_path = parquet_path
        self.table_name = "traits"

        if parquet_path and os.path.exists(parquet_path):
            self._register(parquet_path)
        else:
            # Generate mock data if no file provided
            print("⚠️  No parquet file found — generating mock trait data.")
            self._generate_mock()

    def _register(self, path: str):
        self.con.execute(f"""
            CREATE OR REPLACE VIEW {self.table_name} AS
            SELECT * FROM read_parquet('{path}')
        """)
        count = self.con.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[0]
        print(f"✅ Loaded {count:,} rows from {path}")

    def _generate_mock(self):
        """Generate mock trait data for testing without real parquet file."""
        import numpy as np
        np.random.seed(42)
        n = 50000
        df = pd.DataFrame({
            "trait_id":        [f"T{i:07d}" for i in range(n)],
            "entity_type":     np.random.choice(["simulation","material","molecule","gene"], n),
            "temperature_max": np.random.uniform(20, 1500, n),
            "strength":        np.random.uniform(10, 2000, n),
            "conductivity":    np.random.uniform(0.1, 200, n),
            "ph":              np.random.uniform(2, 12, n),
            "salinity":        np.random.uniform(0, 50, n),
            "source":          np.random.choice(["OpenFOAM","AFLOW","ChEMBL","PDB"], n),
            "location":        np.random.choice(["UP","Delhi","Maharashtra","Punjab"], n),
        })
        self.con.execute(f"CREATE OR REPLACE TABLE {self.table_name} AS SELECT * FROM df")
        print(f"✅ Mock dataset ready: {n:,} traits")

    # ── PUBLIC API ─────────────────────────────────────────────────────────────

    def schema(self) -> pd.DataFrame:
        """Return column names and types."""
        return self.con.execute(f"DESCRIBE {self.table_name}").df()

    def count(self) -> int:
        return self.con.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[0]

    def sample(self, n: int = 5) -> pd.DataFrame:
        return self.con.execute(f"SELECT * FROM {self.table_name} LIMIT {n}").df()

    def filter(self, where_clause: str) -> pd.DataFrame:
        """
        Filter rows with a SQL WHERE clause.
        Example: loader.filter("temperature_max > 500 AND ph < 7")
        """
        sql = f"SELECT * FROM {self.table_name} WHERE {where_clause}"
        return self.con.execute(sql).df()

    def get_batches(self, batch_size: int = 10000, where_clause: str = None):
        """
        Generator that yields DataFrames of batch_size rows.
        Plug into batch_simulator.py.
        """
        total = self.count()
        where = f"WHERE {where_clause}" if where_clause else ""
        offset = 0
        batch_num = 0
        while offset < total:
            sql = f"""
                SELECT * FROM {self.table_name}
                {where}
                LIMIT {batch_size} OFFSET {offset}
            """
            df = self.con.execute(sql).df()
            if df.empty:
                break
            batch_num += 1
            yield batch_num, df
            offset += batch_size

    def load_all(self, where_clause: str = None) -> pd.DataFrame:
        """Load entire dataset (or filtered subset) at once."""
        t0 = time.perf_counter()
        where = f"WHERE {where_clause}" if where_clause else ""
        df = self.con.execute(f"SELECT * FROM {self.table_name} {where}").df()
        elapsed = time.perf_counter() - t0
        print(f"✅ Loaded {len(df):,} rows in {elapsed:.2f}s")
        return df

    def speed_test(self, n: int = 100000) -> dict:
        """Benchmark loading speed."""
        t0 = time.perf_counter()
        df = self.con.execute(
            f"SELECT * FROM {self.table_name} LIMIT {n}"
        ).df()
        elapsed = time.perf_counter() - t0
        rows_per_sec = len(df) / elapsed if elapsed > 0 else 0
        result = {
            "rows_loaded":   len(df),
            "elapsed_sec":   round(elapsed, 3),
            "rows_per_sec":  round(rows_per_sec, 0),
            "est_1M_sec":    round(1_000_000 / rows_per_sec, 1) if rows_per_sec else "N/A",
            "target_met":    elapsed < 30 or rows_per_sec > 33333,
        }
        print(f"\n── Speed Test ────────────────────────────────────")
        print(f"  Rows loaded   : {result['rows_loaded']:,}")
        print(f"  Time          : {result['elapsed_sec']}s")
        print(f"  Speed         : {result['rows_per_sec']:,.0f} rows/sec")
        print(f"  Est. 1M rows  : {result['est_1M_sec']}s  (target: <30s)")
        print(f"  Target met    : {'✅' if result['target_met'] else '❌'}")
        return result


# ── PLUG IN SRIKAR'S PARQUET ──────────────────────────────────────────────────
def load_from_srikar(parquet_path: str) -> DataLoader:
    """
    Call this when Srikar hands over his output parquet.
    Example:
        loader = load_from_srikar("data/raw/real_data_combined.parquet")
    """
    return DataLoader(parquet_path)


if __name__ == "__main__":
    # Test with mock data (replace path with real file when available)
    loader = DataLoader()

    print("\n── Schema ────────────────────────────────────────")
    print(loader.schema().to_string(index=False))

    print(f"\n── Sample rows ───────────────────────────────────")
    print(loader.sample(3).to_string(index=False))

    print(f"\n── Filter test ───────────────────────────────────")
    hot = loader.filter("temperature_max > 1000")
    print(f"  Rows with temp > 1000°C: {len(hot):,}")

    loader.speed_test(50000)
