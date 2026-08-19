import requests
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta

PROMETHEUS_URL = "http://localhost:9090"


def get_historical_metric(query, hours=1):

    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)

    response = requests.get(
        PROMETHEUS_URL + "/api/v1/query_range",
        params={
            "query": query,
            "start": start_time.timestamp(),
            "end": end_time.timestamp(),
            "step": 60
        },
        timeout=10
    )

    response.raise_for_status()

    result = response.json()["data"]["result"]

    if not result:
        raise RuntimeError(
            "No data returned from Prometheus."
        )

    # Take the first returned time series
    values = result[0]["values"]

    data = []

    for timestamp, value in values:
        data.append({
            "Timestamp": datetime.fromtimestamp(float(timestamp)),
            "Value": float(value)
        })

    return pd.DataFrame(data)


# ---------------------------------------------------
# PROMETHEUS QUERIES
# ---------------------------------------------------

cpu_query = '''
100 - (
    avg(
        rate(
            windows_cpu_time_total{mode="idle"}[5m]
        )
    ) * 100
)
'''

memory_query = '''
100 * (
    1 -
    (
        windows_memory_available_bytes
        /
        windows_memory_physical_total_bytes
    )
)
'''


# ---------------------------------------------------
# GET HISTORICAL DATA
# ---------------------------------------------------

print("Getting CPU data...")

cpu_data = get_historical_metric(
    cpu_query,
    hours=1
)

print("Getting Memory data...")

memory_data = get_historical_metric(
    memory_query,
    hours=1
)


# ---------------------------------------------------
# RENAME COLUMNS
# ---------------------------------------------------

cpu_data = cpu_data.rename(
    columns={"Value": "CPU"}
)

memory_data = memory_data.rename(
    columns={"Value": "Memory"}
)


# ---------------------------------------------------
# MERGE USING TIMESTAMP
# ---------------------------------------------------

data = pd.merge(
    cpu_data,
    memory_data,
    on="Timestamp",
    how="inner"
)


# ---------------------------------------------------
# REMOVE MISSING VALUES
# ---------------------------------------------------

data = data.dropna(
    subset=["CPU", "Memory"]
)


# ---------------------------------------------------
# ISOLATION FOREST
# ---------------------------------------------------

model = IsolationForest(
    contamination=0.05,
    random_state=42
)


data["Anomaly"] = model.fit_predict(
    data[["CPU", "Memory"]]
)


# ---------------------------------------------------
# STATUS
# ---------------------------------------------------

data["Status"] = data["Anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})


# ---------------------------------------------------
# DISPLAY RESULT
# ---------------------------------------------------

print("\nAnomaly Detection Result:")

print(data.to_string(index=False))


# ---------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------

data.to_csv(
    "anomaly_report.csv",
    index=False
)


# ---------------------------------------------------
# CURRENT STATUS
# ---------------------------------------------------

print(
    "\nCurrent System Status:",
    data.iloc[-1]["Status"]
)

print(
    "\nReport generated: anomaly_report.csv"
)
