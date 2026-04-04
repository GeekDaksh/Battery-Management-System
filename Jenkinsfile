pipeline {
    agent any

    stages {

        stage('Clone Repo') {
            steps {
                git 'https://github.com/YOUR_USERNAME/YOUR_REPO.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t bms-ai .'
            }
        }

        stage('Stop Old Container') {
            steps {
                sh 'docker stop bms-container || true'
                sh 'docker rm bms-container || true'
            }
        }

        stage('Run Container') {
            steps {
                sh 'docker run -d -p 8000:8000 --name bms-container bms-ai'
            }
        }

    }
}