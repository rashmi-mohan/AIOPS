import requests
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta


# ============================================================
# CONFIGURATION
# ============================================================

PROMETHEUS_URL = "http://localhost:9090"

# Collect data for the last 1 hour
HISTORY_HOURS = 1

# Prometheus sampling interval
STEP = "60s"


# ============================================================
# FUNCTION: GET HISTORICAL METRIC FROM PROMETHEUS
# ============================================================

def get_metric_history(query, start_time, end_time):

    print("\nExecuting Prometheus query:")
    print(query.strip())

    response = requests.get(
        PROMETHEUS_URL + "/api/v1/query_range",
        params={
            "query": query,
            "start": start_time.timestamp(),
            "end": end_time.timestamp(),
            "step": STEP
        },
        timeout=10
    )

    # Check HTTP error
    response.raise_for_status()

    # Convert response to JSON
    result = response.json()["data"]["result"]

    # Check whether Prometheus returned data
    if not result:
        raise RuntimeError(
            "No data returned from Prometheus."
        )

    # Take first returned time series
    values = result[0]["values"]

    rows = []

    for timestamp, value in values:

        rows.append({
            "Timestamp": datetime.fromtimestamp(
                float(timestamp)
            ),
            "Value": float(value)
        })

    return pd.DataFrame(rows)


# ============================================================
# PROMETHEUS QUERIES
# ============================================================

# CPU utilization
cpu_query = """
100 -
(
    avg(
        rate(
            windows_cpu_time_total{mode="idle"}[5m]
        )
    ) * 100
)
"""


# Memory utilization
memory_query = """
100 *
(
    1 -
    (
        windows_memory_available_bytes
        /
        windows_memory_physical_total_bytes
    )
)
"""


# ============================================================
# MAIN PROGRAM
# ============================================================

try:

    print("=" * 60)
    print("AIOps Anomaly Detection")
    print("=" * 60)

    # --------------------------------------------------------
    # CREATE ONE COMMON TIME WINDOW
    # --------------------------------------------------------

    end_time = datetime.now()

    start_time = (
        end_time -
        timedelta(hours=HISTORY_HOURS)
    )

    print("\nTime range:")
    print("Start:", start_time)
    print("End  :", end_time)


    # --------------------------------------------------------
    # GET CPU DATA
    # --------------------------------------------------------

    print("\nGetting CPU data...")

    cpu_data = get_metric_history(
        cpu_query,
        start_time,
        end_time
    )


    # --------------------------------------------------------
    # GET MEMORY DATA
    # --------------------------------------------------------

    print("\nGetting Memory data...")

    memory_data = get_metric_history(
        memory_query,
        start_time,
        end_time
    )


    # --------------------------------------------------------
    # RENAME COLUMNS
    # --------------------------------------------------------

    cpu_data.rename(
        columns={
            "Value": "CPU"
        },
        inplace=True
    )

    memory_data.rename(
        columns={
            "Value": "Memory"
        },
        inplace=True
    )


    # --------------------------------------------------------
    # DISPLAY NUMBER OF RECORDS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATA INFORMATION")
    print("=" * 60)

    print(
        "CPU rows    :",
        len(cpu_data)
    )

    print(
        "Memory rows :",
        len(memory_data)
    )


    # --------------------------------------------------------
    # DISPLAY CPU TIME RANGE
    # --------------------------------------------------------

    print("\nCPU time range:")

    print(
        "Start:",
        cpu_data["Timestamp"].min()
    )

    print(
        "End  :",
        cpu_data["Timestamp"].max()
    )


    # --------------------------------------------------------
    # DISPLAY MEMORY TIME RANGE
    # --------------------------------------------------------

    print("\nMemory time range:")

    print(
        "Start:",
        memory_data["Timestamp"].min()
    )

    print(
        "End  :",
        memory_data["Timestamp"].max()
    )


    # --------------------------------------------------------
    # SORT DATA
    # --------------------------------------------------------

    cpu_data = cpu_data.sort_values(
        "Timestamp"
    )

    memory_data = memory_data.sort_values(
        "Timestamp"
    )


    # --------------------------------------------------------
    # MERGE CPU AND MEMORY
    # --------------------------------------------------------
    #
    # merge_asof() matches the nearest timestamp.
    # This handles small differences such as:
    #
    # CPU    09:05:48.450
    # Memory 09:05:48.514
    #
    # --------------------------------------------------------

    data = pd.merge_asof(
        cpu_data,
        memory_data,
        on="Timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("30s")
    )


    # --------------------------------------------------------
    # REMOVE MISSING VALUES
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            "CPU",
            "Memory"
        ]
    )


    # --------------------------------------------------------
    # DISPLAY MERGED DATA INFORMATION
    # --------------------------------------------------------

    print("\nRows after CPU/Memory merge:")

    print(len(data))


    # --------------------------------------------------------
    # CHECK WHETHER DATA EXISTS
    # --------------------------------------------------------

    if data.empty:

        raise RuntimeError(
            "\nNo matching CPU and Memory data found.\n"
            "Check Prometheus and Windows Exporter."
        )


    # --------------------------------------------------------
    # CHECK MINIMUM DATA
    # --------------------------------------------------------

    if len(data) < 10:

        raise RuntimeError(
            f"\nNot enough data for Isolation Forest.\n"
            f"Only {len(data)} matching samples available.\n"
            f"At least 10 samples are required."
        )


    # --------------------------------------------------------
    # DISPLAY FIRST 10 RECORDS
    # --------------------------------------------------------

    print("\nCombined CPU + Memory data:")

    print(
        data.head(10).to_string(
            index=False
        )
    )


    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    print("\n" + "=" * 60)
    print("Training Isolation Forest...")
    print("=" * 60)


    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )


    # --------------------------------------------------------
    # TRAIN AND PREDICT
    # --------------------------------------------------------

    data["Anomaly"] = model.fit_predict(
        data[
            [
                "CPU",
                "Memory"
            ]
        ]
    )


    # ========================================================
    # CONVERT RESULT
    # ========================================================

    data["Status"] = data["Anomaly"].map(
        {
            1: "Normal",
            -1: "Anomaly"
        }
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n" + "=" * 60)
    print("ANOMALY DETECTION RESULT")
    print("=" * 60)

    print(
        data.to_string(
            index=False
        )
    )


    # ========================================================
    # SAVE CSV REPORT
    # ========================================================

    report_file = "anomaly_report.csv"

    data.to_csv(
        report_file,
        index=False
    )


    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60)

    print(
        "Report generated:",
        report_file
    )


    # ========================================================
    # CURRENT SYSTEM STATUS
    # ========================================================

    current_status = data.iloc[-1]["Status"]

    current_cpu = data.iloc[-1]["CPU"]

    current_memory = data.iloc[-1]["Memory"]


    print("\nCurrent CPU Usage    :",
          round(current_cpu, 2), "%")

    print("Current Memory Usage :",
          round(current_memory, 2), "%")

    print(
        "Current System Status:",
        current_status
    )


    # ========================================================
    # COUNT ANOMALIES
    # ========================================================

    anomaly_count = (
        data["Anomaly"] == -1
    ).sum()

    normal_count = (
        data["Anomaly"] == 1
    ).sum()


    print("\nNormal samples  :", normal_count)

    print("Anomaly samples :", anomaly_count)


    # ========================================================
    # FINAL RESULT
    # ========================================================

    if current_status == "Anomaly":

        print("\nWARNING: CURRENT SYSTEM IS ANOMALOUS!")

    else:

        print("\nSystem is operating normally.")


    print("\nAIOps anomaly detection completed successfully.")


# ============================================================
# ERROR HANDLING
# ============================================================

except requests.exceptions.ConnectionError:

    print("\nERROR: Cannot connect to Prometheus.")

    print(
        "Make sure Prometheus is running at:"
    )

    print(PROMETHEUS_URL)

    raise


except requests.exceptions.Timeout:

    print(
        "\nERROR: Prometheus request timed out."
    )

    raise


except requests.exceptions.HTTPError as e:

    print(
        "\nERROR: Prometheus returned an HTTP error:"
    )

    print(e)

    raise


except Exception as e:

    print(
        "\nERROR in AIOps anomaly detection:"
    )

    print(e)

    raise
