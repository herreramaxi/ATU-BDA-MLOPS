# ATU-BDA-MLOPS

## 📂 Git Branching Strategy (GitFlow)
This project follows the [GitFlow](https://nvie.com/posts/a-successful-git-branching-model/) branching strategy to manage development and releases in a structured and scalable way.

### 🔀 Branch Types
#### main
This branch contains production-ready code. All completed features and hotfixes are merged here after passing all tests.

#### develop
This branch serves as the integration branch for features. New work is based on this branch and merged back into it before releasing.

#### feature/*
Each new feature or experiment is developed in a dedicated feature/<name> branch. These are branched from develop and merged back into it upon completion.

#### release/*
Used to prepare a new production release. These are branched from develop, finalized, then merged into both main and develop.

#### hotfix/*
For urgent fixes to production. These are branched from main and merged back into both main and develop.
