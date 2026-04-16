# Responsible AI in Hiring Systems: A Temporal Analysis of Bias and Fairness (2014–2022)

## Overview
This project analyzes how bias in AI-driven income prediction (a proxy for hiring decisions) has evolved over time using U.S. Census data from 2014, 2018, and 2022. We train Random Forest classifiers, evaluate fairness across gender and race, apply bias mitigation, and analyze trade-offs between accuracy and fairness.

## Team Members
- Patnam Jahnavi [S20230010181]
- Sahal Ansar [S20230010210]
- Venkat Rahul [S20230010257]

## Dataset
- **Source:** American Community Survey (ACS) Public Use Microdata Sample via the `folktables` Python package
- **Years:** 2014, 2018, 2022
- **States:** California, New York, Texas
- **Size:** ~411K (2014), ~435K (2018), ~453K (2022) records
- **Task:** Binary classification — predict whether annual income exceeds $50,000
- **Note:** Data is downloaded automatically when you run the notebook. No manual download needed.

## Project Structure
```
├── ResAI_Project.ipynb        # Main notebook (all code + outputs)
├── report.pdf                 # Final project report
├── README.md                  # This file
├── .gitignore
├── requirements.txt
├── gender_income_gap.png
├── race_distribution.png
├── fairness_trends.png
├── before_after_mitigation.png
├── feature_importance_comparison.png
├── individual_explanation.png
├── accuracy_vs_fairness_tradeoff.png
├── race_selection_rate.png
├── gender_vs_race_mitigation.png
├── intersectional_gender_education.png
└── data/                      # Auto-downloaded (gitignored)
```

## How to Run

### 1. Clone the repository
```bash
git clone <repo-url>
cd <repo-folder>
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebook
```bash
jupyter notebook ResAI_Project.ipynb
```
Run all cells sequentially. The first run will download ~50MB of Census data (3 states x 3 years) into a `data/` folder. Subsequent runs will use cached data.

### 5. Expected runtime
| Step | Time |
|------|------|
| Data download (first run only) | 2-3 minutes |
| EDA and visualizations | < 30 seconds |
| Model training (3 years) | 1-2 minutes |
| Fairness analysis + mitigation | < 1 minute |
| Feature importance + explanations | < 30 seconds |
| **Total (first run)** | **~5 minutes** |

## Tools and Libraries
- **Data:** `folktables` (Census data access)
- **ML:** `scikit-learn` (Random Forest classifier)
- **Fairness:** `fairlearn` (metrics + ThresholdOptimizer mitigation)
- **Explainability:** `shap`, Gini feature importance
- **Visualization:** `matplotlib`, `seaborn`

## Key Results
- Gender bias (DPD) decreased from 0.165 (2014) to 0.147 (2022), but remains above the 0.05 fairness threshold
- Racial bias (DPD) worsened from 0.325 to 0.362 despite increasing workforce diversity
- Post-processing mitigation reduced gender EOD to 0.004 at less than 1% accuracy cost
- Race mitigation was less effective (EOD: 0.09 after mitigation) and costlier (2.12% accuracy)

## References
1. Ding et al. (2021). Retiring Adult: New Datasets for Fair Machine Learning. NeurIPS.
2. Fabris et al. (2024). Fairness and Bias in Algorithmic Hiring. ACM TIST.
3. Dastin (2018). Amazon scraps secret AI recruiting tool. Reuters.
4. Bird et al. (2020). Fairlearn: Assessing and improving fairness in AI. Microsoft.