# Customer Churn Prediction — Lloyds Banking Group (Forage Project)

This project focuses on predicting customer churn using supervised machine learning techniques. The work was completed as part of a **Lloyds Banking Group virtual experience program on Forage**, simulating a real-world data science and analytics workflow in the financial services sector.

The objective is to identify customers at risk of churning and to evaluate models that balance predictive performance with interpretability for business decision-making.

## Structure

- `notebooks/` — Jupyter notebook containing analysis, feature engineering, and modeling
- `data/` — local dataset files (not tracked in GitHub)
- `.gitignore` — excludes datasets and environment-specific files

## Project Context

This project was completed through the **Lloyds Banking Group Forage job simulation**, which mirrors common industry tasks such as:
- Exploratory data analysis
- Feature engineering and selection
- Model training and evaluation
- Interpreting results in a business context

## Data

The dataset used in this project was provided as part of the Lloyds Banking Group Forage simulation.  
The data may have been **anonymized or modified** for instructional purposes and is **not included** in this repository.

## How to Run

This repository is intended for review of methodology, modeling choices, and interpretation rather than exact numerical replication.

To run locally:
1. Place the dataset CSV file in the `data/` directory
2. Open the notebook in `notebooks/`
3. Run the notebook top to bottom

Example local data path:
```python
pd.read_csv("../data/churn.csv")
