pipeline {
    agent any
    stages {
        stage('Download Data') {
            steps {
                build job: 'download'
            }
        }
        stage('Train Model') {
            steps {
                build job: 'train_model'
            }
        }
        stage('Deploy Model') {
            steps {
                build job: 'deploy'
            }
        }
        stage('Check Status') {
            steps {
                build job: 'healthy'
            }
        }
    }
}