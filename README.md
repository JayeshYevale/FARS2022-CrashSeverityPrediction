# Identifying Crash Patterns for Injury Severity and Death Occurrence Using FARS (2022)

**Course:** IS 597 MLC — Machine Learning Cloud
**Author:** Jayesh Yevale

## Overview

This project analyzes 2022 crash data from the **Fatality Analysis Reporting System (FARS)**, a nationwide census maintained by the National Highway Traffic Safety Administration (NHTSA) that records fatal motor vehicle crashes across the U.S. The goal is to identify patterns behind crash **injury severity** and **death occurrence**, and to compare several machine learning models for predicting these outcomes — surfacing findings that could help raise public awareness of high-risk conditions, beyond the government's own descriptive reporting.

## Motivation

FARS publishes detailed statistics, but there's little analysis from the driver's-eye perspective: which conditions and circumstances are most predictive of severe injury or death. This project applies supervised ML models to the FARS 2022 data to explore that gap and identify the most reliable predictors and best-performing model for each target.

## Data

* **Source:** [NHTSA FARS](https://www.nhtsa.gov/), U.S. Department of Transportation
* **Files used:** Accident (\~39K rows, 80 attributes), Person (\~96K rows, 126 attributes), and Vehicle (\~60K rows, 201 attributes) CSVs, joined on relevant keys after attribute selection
* **Targets:**

  * `A\_PERINJ` — injury severity
  * `A\_DOA` — death occurrence
* Key predictors include crash event type, collision direction, vehicle body type, route type, locality (rural/urban), lighting conditions, airbag deployment, drinking status, and more.

## Methodology

1. **Preprocessing** — attribute selection, join across Accident/Person/Vehicle tables, binary recoding of categorical targets, feature scaling (`StandardScaler`), train/test split.
2. **Modeling** — trained and compared four classifiers:

   * Decision Tree
   * Logistic Regression (cross-validated)
   * Random Forest
   * Gradient Boosting
3. **Evaluation** — Precision, Recall, F1-score, and Accuracy per class.

## Results

|Model|Class|Precision|Recall|F1|Accuracy|
|-|-|-|-|-|-|
|Logistic Regression|Deaths (1)|0.9318|0.9140|0.9228|0.885|
|Logistic Regression|No Deaths (0)|0.7529|0.7967|0.7742||
|Decision Tree|Deaths (1)|0.9902|0.7902|0.8789|0.8362|
|Decision Tree|No Deaths (0)|0.6048|0.9761|0.7468||
|Random Forest|Deaths (1)|0.8122|0.9785|0.8877|0.8136|
|Random Forest|No Deaths (0)|0.8269|0.3122|0.4533||
|**Gradient Boosting**|Deaths (1)|0.9395|0.9164|0.9278|**0.8927**|
|**Gradient Boosting**|No Deaths (0)|0.7635|0.8205|0.7910||

**Gradient Boosting** performed best overall, followed by **Logistic Regression** — both showing strong precision/recall balance on predicting fatal outcomes.

## Repository Structure

```
├── FARS2022\_Person\_Accident.xlsx           # Preprocessed source dataset
├── IS597MLC\_Final\_Project\_Yevale\_Jayesh.ipynb   # Main analysis notebook
├── IS597MLC\_Final\_Project\_Yevale\_Jayesh.pdf     # Full written project report
├── modules.py                              # Reusable functions: data loading/EDA, model fitting, evaluation
└── README.md
```

## How to Run

1. Clone the repo and install dependencies:

```
   pip install numpy pandas scikit-learn matplotlib seaborn nltk openpyxl
   ```

2. Open `IS597MLC\_Final\_Project\_Yevale\_Jayesh.ipynb` in Jupyter.
3. The notebook uses helper functions from `modules.py`:

   * `load\_data(in\_filename, target)` — loads and cleans the dataset, generates EDA plots (class distribution, feature distributions, correlation heatmap), and returns scaled train/test splits.
   * `fit\_model(X\_train, y\_train, model)` — trains one of `'DT'`, `'SVM'`, `'LR'`, `'RF'`, `'GB'`.
   * `evaluate\_model(y\_test, y\_pred, eval\_model)` — prints confusion matrix and classification report.

## Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn

## Full Report

See [`IS597MLC\_Final\_Project\_Yevale\_Jayesh.pdf`](./IS597MLC_Final_Project_Yevale_Jayesh.pdf) for the complete write-up, including literature review and detailed data dictionary.

## License

This project is licensed under the \[MIT License](LICENSE).

