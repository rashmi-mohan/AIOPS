import requests
import pandas as pd
from sklearn.ensemble import IsolationForest

PROMETHEUS_URL = "http://localhost:9090"

def get_metric(query):
    response = requests.get(PROMETHEUS_URL + "/api/v1/query", params={"query": query}, timeout=10)
    response.raise_for_status()
    values = response.json()["data"]["result"]
    print("Prometheus response:", response.text)

    if not values:
        raise RuntimeError("No data returned from Prometheus. Check Prometheus and Windows Exporter.")
    return float(values[0]["value"][1])

cpu_query = '100 - (avg(rate(windows_cpu_time_total{mode="idle"}[5m])) * 100)'
memory_query ='100 * (1 - (windows_memory_available_bytes /windows_memory_physical_total_bytes))'
cpu = get_metric(cpu_query)
memory = get_metric(memory_query)
print("CPU Usage    :", round(cpu, 2), "%")
print("Memory Usage :", round(memory, 2), "%")

historical = pd.DataFrame({
    "CPU": [40, 45, 50, 48, 52, 55, 51, 49, 47],
    "Memory": [45, 48, 50, 52, 49, 51, 53, 50, 48]
})
current = pd.DataFrame({"CPU": [cpu], "Memory": [memory]})
data = pd.concat([historical, current], ignore_index=True)

model = IsolationForest(contamination=0.1, random_state=42)
data["Anomaly"] = model.fit_predict(data[["CPU", "Memory"]])
data["Status"] = data["Anomaly"].map({1: "Normal", -1: "Anomaly"})

print("\nAnomaly Detection Result:")
print(data)
data.to_csv("anomaly_report.csv", index=False)
print("\nCurrent System Status:", data.iloc[-1]["Status"])
print("Report generated: anomaly_report.csv")
