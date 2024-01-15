# OPTIVER: Trading at the Close

## Overview
This repository contains the code and a written report for the "Optiver Trading at the Close" Kaggle Code Competition. The competition details and dataset can be found [here](https://www.kaggle.com/competitions/optiver-trading-at-the-close/data).

## Using MLFlow

MLFlow is an open-source platform for managing the end-to-end machine learning lifecycle. For this particular instance we are interested in Experiment Tracking, keeping all models neatly stored, to compare their performance and also be able to reproduce experiments.

To use MLFlow in this project, you can access the MLFlow User Interface (UI) with the following command, after changing to this directory:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

This command starts the MLFlow tracking server with the UI. The `--backend-store-uri` option specifies the location of the database where MLFlow will store its metadata. In this case, it's a SQLite database (`mlflow.db`) in the current directory. This server allows you to visually interact with the MLFlow Experiments, Runs, and Artifacts.

## Project Description



Feel free to explore the code and report for more details on the project. If you have any questions or feedback, please don't hesitate to reach out.
