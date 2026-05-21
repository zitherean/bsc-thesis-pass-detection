# Forward Football Thesis Project

This repository contains the code and notebooks for my thesis project.

## Branches

- `main` - final branch. Use this branch for the completed implementation and final results.
- `test` - used for experimenting with and testing new ideas.
- `final` - contains the implemented idea.

> Use branch `main`.

## Repository Structure

- `data_matches/` - match data organized by match folders including processed matches saved as joblib files.
- `notebooks/` - Jupyter notebooks for testing and evaluation.
- `plots/` - generated plots and visualizations.
- `saved_models/` - saved models to reproduce test results.
- `src/` - source code for data processing, feature engineering, models, and pipeline logic.

## Notebooks

There are three notebooks under `notebooks/`:

1. `playground.ipynb` - run this for a quick test on only 1 match.
2. `pass_detection.ipynb` - run this to test on all matches.
3. `model_test.ipynb` - reproduce test results using existing saved models.

## Recommended Usage

- For a quick test, open and run `notebooks/playground.ipynb`.
- For a full pass detection test across all matches, open and run `notebooks/pass_detection.ipynb`. Running logistic regression should not take longer than 5 minutes to train and test.
- To reproduce test results from existing models, open and run `notebooks/model_test.ipynb`.
- `pass_detection.ipynb` and `model_test.ipynb` depend on `.joblib` files and should not cause issues. If for some reason the `.joblib` files are not present the full model pipeliine will be run, which can take 1 hour. If this happens run `playground.ipynb` instead.

## Notes for Users

If you want to use this project:

- Clone the repository.
- Switch to `main` branch.
- Create a virtual environment:
  - `python -m venv venv`
- Install dependencies:
  - `python -m pip install -r requirements.txt`
- Run one of the notebooks under `notebooks/`.
