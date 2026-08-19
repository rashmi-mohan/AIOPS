import requests
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta

PROMETHEUS_URL = "http://localhost:9090"


def get_metric_history(query, hours=1):

    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)

    response = requests.get(
        PROMETHEUS_URL + "/api/v1/query_range",
        params={
            "query": query,
            "start": start_time.timestamp(),
            "end": end_time.timestamp(),
            "step": "60s"
        },
        timeout=10
    )

    response.raise_for_status()

    result = response.json()["data"]["result"]

    if not result:
        raise RuntimeError(
            "No data returned from Prometheus for query:\n" + query
        )

    values = result[0]["values"]

    rows = []

    for timestamp, value in values:

        rows.append({
            "Timestamp": datetime.fromtimestamp(float(timestamp)),
            "Value": float(value)
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# CPU QUERY
# --------------------------------------------------

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


# --------------------------------------------------
# MEMORY QUERY
# --------------------------------------------------

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


# --------------------------------------------------
# GET DATA
# --------------------------------------------------

print("Getting CPU data...")

cpu_data = get_metric_history(
    cpu_query,
    hours=1
)

print("Getting Memory data...")

memory_data = get_metric_history(
    memory_query,
    hours=1
)


# --------------------------------------------------
# RENAME
# --------------------------------------------------

cpu_data.rename(
    columns={"Value": "CPU"},
    inplace=True
)

memory_data.rename(
    columns={"Value": "Memory"},
    inplace=True
)


# --------------------------------------------------
# DEBUG
# --------------------------------------------------

print("\nCPU rows:", len(cpu_data))
print("Memory rows:", len(memory_data))

print("\nCPU sample:")
print(cpu_data.head())

print("\nMemory sample:")
print(memory_data.head())


# --------------------------------------------------
# MERGE
# --------------------------------------------------

data = pd.merge(
    cpu_data,
    memory_data,
    on="Timestamp",
    how="inner"
)


print("\nRows after merge:", len(data))


# --------------------------------------------------
# CHECK EMPTY DATA
# --------------------------------------------------

if data.empty:

    print("\nERROR: CPU and Memory timestamps do not match.")

    print("\nCPU timestamps:")
    print(cpu_data["Timestamp"].head())

    print("\nMemory timestamps:")
    print(memory_data["Timestamp"].head())

    raise RuntimeError(
        "No common timestamps between CPU and Memory data."
    )


# --------------------------------------------------
# REMOVE INVALID VALUES
# --------------------------------------------------

data = data.dropna(
    subset=["CPU", "Memory"]
)


if data.empty:

    raise RuntimeError(
        "No valid CPU/Memory samples available after removing missing values."
    )


# --------------------------------------------------
# ISOLATION FOREST
# --------------------------------------------------

print("\nTraining Isolation Forest...")

model = IsolationForest(
    contamination=0.05,
    random_state=42
)


data["Anomaly"] = model.fit_predict(
    data[["CPU", "Memory"]]
)


# --------------------------------------------------
# STATUS
# --------------------------------------------------

data["Status"] = data["Anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print("\nAnomaly Detection Result:")

print(
    data.to_string(index=False)
)


# --------------------------------------------------
# SAVE REPORT
# --------------------------------------------------

data.to_csv(
    "anomaly_report.csv",
    index=False
)


# --------------------------------------------------
# CURRENT STATUS
# --------------------------------------------------

current_status = data.iloc[-1]["Status"]

print(
    "\nCurrent System Status:",
    current_status
)

print(
    "\nReport generated: anomaly_report.csv"
)
