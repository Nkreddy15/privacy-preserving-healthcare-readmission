# Privacy-Preserving Healthcare Readmission Prediction

This project investigates bias in healthcare AI systems while protecting patient privacy through differential privacy techniques. It predicts 30-day hospital readmission using the UCI diabetic readmission dataset and evaluates fairness across racial groups.

## Problem Statement

Healthcare prediction models can inherit bias from historical medical data. This project examines whether a readmission prediction model produces unequal outcomes across racial groups while applying privacy-preserving machine learning techniques.

## Features

- Hospital readmission prediction
- Differential privacy using IBM `diffprivlib`
- Class balancing to reduce target imbalance
- Race-based prediction analysis
- Fairness-oriented evaluation
- Ethical AI report and presentation

## Technology Stack

- Python
- Pandas
- Scikit-Learn
- diffprivlib
- AIF360 concepts
- Matplotlib
- Jupyter Notebook

## Reported Results

From the project report:

- Accuracy: 71.3%
- Precision for readmitted class: 70%
- Recall for readmitted class: 68%
- Disparate Impact: 0.73
- Equal Opportunity Difference: -0.11

## Repository Structure

```text
.
├── data/
├── notebooks/
├── src/
├── reports/
├── screenshots/
├── docs/
├── README.md
└── requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
python src/healthcare_readmission_bias_analysis.py
```

Or open the notebook:

```text
notebooks/healthcare_readmission_bias_analysis.ipynb
```

## Ethical Note

This project is for educational and research demonstration purposes. It should not be used for clinical decision-making.

## Author

Nikhil Kumar Reddy Chalamalla
