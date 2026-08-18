# AIOps Anomaly Detection Using Prometheus and Isolation Forest

## Aim
To collect system performance metrics from Prometheus and detect anomalies using the Isolation Forest algorithm, integrated with Jenkins.

## Architecture
Windows Exporter -> Prometheus -> Python -> Isolation Forest -> Anomaly Report -> Jenkins

## Requirements
- Python 3.x
- Prometheus
- Windows Exporter
- Jenkins
- Git/GitHub

## Setup
1. Install Windows Exporter. Verify: http://localhost:9182/metrics
2. Configure Prometheus using prometheus.yml.
3. Start Prometheus and open http://localhost:9090.
4. Verify Windows metrics are available.
5. Install Python dependencies: python -m pip install -r requirements.txt
6. Run: python anomaly_detection.py
7. Create a Jenkins Pipeline using the Jenkinsfile.
8. Click Build Now.

## Output
The program generates anomaly_report.csv.
