import yaml
from pathlib import Path
import pandas as pd
import duckdb

def _get_nvtx_events(con): 
    # Use sqlite_scan to read the table directly from the file
    # This keeps memory usage low and speed high
    query = f"SELECT * FROM sqlite_db.NVTX_EVENTS"
    
    # .df() converts the DuckDB relation to a Pandas DataFrame
    return con.sql(query).df()

def _get_cpu_kernel_events(con, start_time, end_time): 
    cpu_query = f"""
    SELECT 
        c.start AS c_start, 
        c.end AS c_end, 
        k.start AS k_start, 
        k.end AS k_end, 
        k.deviceId
    FROM sqlite_db.CUPTI_ACTIVITY_KIND_RUNTIME AS c
    INNER JOIN sqlite_db.CUPTI_ACTIVITY_KIND_KERNEL AS k
        ON c.correlationId = k.correlationId
    WHERE c.start >= {start_time} 
      AND c.end <= {end_time}
    """
    return con.sql(cpu_query).df()

def _get_cpu_starts(con, nvtx_events, cpu_events, tolerance=100000):
    
    query = f"""
    WITH filtered_nvtx AS (
        SELECT * FROM nvtx_events 
        WHERE text LIKE '%''model.model''%'
    )
    SELECT 
        n.*, 
        c.c_start, c.c_end, c.k_start, c.k_end, c.deviceId
    FROM filtered_nvtx n
    ASOF JOIN cpu_events c
      ON n.domainId = c.deviceId   -- Equivalent to left_by/right_by
      AND n.start <= c.c_start     -- Equivalent to direction='forward'
    WHERE (c.c_start - n.start) <= {tolerance}
    ORDER BY c.deviceId, c.k_start
    """
    return con.sql(query).df()

def _get_cpu_ends(con, nvtx_events, cpu_events, tolerance=100000):
    # Check for logits_processor in the nvtx_events DataFrame
    # Note: We can do this check via DuckDB as well for consistency
    has_logits = con.sql("SELECT EXISTS (SELECT 1 FROM nvtx_events WHERE text LIKE '%logits_processor%')").fetchone()[0]
    search_text = 'logits_processor' if has_logits else "''model.model''"

    query = f"""
    WITH filtered_nvtx AS (
        SELECT * FROM nvtx_events 
        WHERE text LIKE '%{search_text}%'
          AND 'end' IS NOT NULL
    )
    SELECT 
        n.*, 
        c.c_start, c.c_end, c.k_start, c.k_end, c.deviceId
    FROM filtered_nvtx n
    ASOF JOIN cpu_events c
      ON n.domainId = c.deviceId
      AND n.end >= c.c_end         -- Equivalent to direction='backward'
    WHERE (n.end - c.c_end) <= {tolerance}
    ORDER BY c.deviceId, c.k_end
    """
    return con.sql(query).df()

def _get_kernel_timespans(con, starts, ends): 
    query = f"""
    SELECT
        s.k_start as k_start,
        e.k_end as k_end,
        e.deviceId as deviceId   
    FROM starts s
    ASOF JOIN ends e
      ON s.deviceId = e.deviceId
      AND s.k_end <= e.k_start         
    ORDER BY e.deviceId, s.k_end
    """
    return con.sql(query).df()

def _get_results_df(db_path):
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{db_path}' AS sqlite_db (TYPE SQLITE, READ_ONLY);")


    nvtx_events = _get_nvtx_events(con)

    stats = con.sql("""
        SELECT MIN("start") as s, MAX("end") as e FROM sqlite_db.NVTX_EVENTS
    """).fetchone()
    stats = con.sql(
    f"""
    SELECT min("start"), max("end") 
    FROM sqlite_db.NVTX_EVENTS
    """
    ).fetchone()

    start_time, end_time = stats
    cpu_events = _get_cpu_kernel_events(con, start_time, end_time)

    starts = _get_cpu_starts(con, nvtx_events, cpu_events)
    ends = _get_cpu_ends(con, nvtx_events, cpu_events)
    time_segments = _get_kernel_timespans(con, starts, ends)

    all_metrics = con.sql("""
    SELECT DISTINCT metricId, metricName
    FROM sqlite_db.TARGET_INFO_GPU_METRICS 
    WHERE metricName LIKE '%NVLink%' or metricName LIKE '%PCIe%'
    """).df()
    # print('\tbeginning merge')
    results = con.sql("""
    SELECT 
        m.metricID,
        f.metricName,
        MIN(m.value) AS min_val,
        MAX(m.value) AS max_val,
        AVG(m.value) AS avg_val,
        MIN(CASE WHEN m.value != 0 THEN m.value END) AS min_no_zero,
        AVG(CASE WHEN m.value != 0 THEN m.value END) AS avg_no_zero,
        COUNT(CASE WHEN m.value != 0 THEN m.value END) as cnt_no_zero,
        MIN(CASE WHEN m.value > 1 THEN m.value END) AS min_gt_one,
        AVG(CASE WHEN m.value > 1 THEN m.value END) AS avg_gt_one, 
        COUNT(CASE WHEN m.value > 1 THEN m.value END) as cnt_gt_one,
    FROM sqlite_db.GPU_METRICS m
    INNER JOIN time_segments e 
        ON (m.typeId & 255) = e.deviceID
        AND m.timestamp >= e.k_start 
        AND m.timestamp <= e.k_end
    INNER JOIN all_metrics f
        ON m.metricID = f.metricID
    GROUP BY m.metricID, f.metricName
    """)

    # print('\tfinished')
    results_df = results.df()
    return results_df

def _get_stats_from_df(df):
    metrics_of_interest = [
        "NVLink RX Responses User Data [Throughput %]",
        "NVLink TX Responses User Data [Throughput %]",
    ]

    columns_to_extract = [
        "avg_val",
        "max_val",
        "avg_gt_one",
        "cnt_gt_one",
        "min_gt_one",
    ]

    def to_float(value, default=0.0):
        if pd.isna(value):
            return default
        return float(value)

    stats = {}

    for metric in metrics_of_interest:
        row = df.loc[df["metricName"] == metric]

        # skip if metric is missing entirely
        if row.empty:
            continue

        row = row.iloc[0]

        stats[metric] = {
            col: to_float(row[col])
            for col in columns_to_extract
        }

    return stats

def metric_cal(directory: str) -> float:
    """
    Calculate the bandwidth utilization for peertopeer from the exported sqlite file from nsys.

    only applicable pp > 1.

    Args:
        directory (str): The directory path containing the exported sqlite file from nsys.

    Returns:
        Dict[str, Dict[str, float]] | "n/a": The statistics of bandwidth utilization for peertopeer, or "n/a" if the metric is not applicable.
    """
    dir_name = Path(directory).name
    db_path = str(Path(directory) / "nsys_0.sqlite")
    workload_card_path = Path(directory) / (dir_name + ".yaml")
    output_csv_path = Path(directory) / "bandwidth_utilization_peertopeer_0.csv"

    # Parse workload card to get metadata
    with open(workload_card_path, 'r') as f:
        workload_card = yaml.safe_load(f)
        model_family = workload_card["workload"]["model"]["model_family"]
        pp = workload_card["Model-executor"]["model_plan_parallelization"]["pp"]

    if model_family not in ["deepseek-v2-lite", "llama-3.1-8B", "qwen-32b"]:
        return "n/a"

    if pp <= 1:
        return "n/a"

    results_df = _get_results_df(db_path)
    results_df.to_csv(output_csv_path)
    stats = _get_stats_from_df(results_df)

    if model_family == "qwen-32b" and pp == 2:
        stats = {"(Node 0) " + k: v for k, v in stats.items()}
        db_path_1 = str(Path(directory) / "nsys_1.sqlite")
        results_df_1 = _get_results_df(db_path_1)
        output_csv_path_1 = Path(directory) / "bandwidth_utilization_peertopeer_1.csv"
        results_df_1.to_csv(output_csv_path_1)
        stats_1 = _get_stats_from_df(results_df_1)
        stats.update({"(Node 1) " + k: v for k, v in stats_1.items()})

    return stats