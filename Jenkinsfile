pipeline {
    agent any

    stages {
        stage('Install Dependencies') {
            steps {
                bat 'python -m pip install -r requirements.txt'
            }
        }

        stage('AIOps Anomaly Detection') {
            steps {
                bat 'python anomaly_detection.py'
            }
        }

        stage('Archive Report') {
            steps {
                archiveArtifacts artifacts: 'anomaly_report.csv', allowEmptyArchive: false
            }
        }
    }

    post {
        success {
            echo 'AIOps anomaly detection completed successfully.'
        }
        failure {
            echo 'AIOps pipeline failed. Check the Jenkins console.'
        }
    }
}
