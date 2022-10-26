# kaggle
open competition experiences in kaggle
## Tabular Playground Series - Sep 2022
Forecast a whole year numbers of selling in six countries and four books.
- Baseline: [[TPS-SEP-22] EDA and Linear Regression Baseline](https://www.kaggle.com/code/cabaxiom/tps-sep-22-eda-and-linear-regression-baseline).
- **My work**: linear regression with feature engineering by week.  

It seems that the baseline version made a better prediction with feature engineering in day of year.  
## Tabular Playground Series - Aug 2022
Predict individual product failures (in probability) of new codes with their individual lab test results.
- Baseline: [Simple Logistic Regression for Good Score (0.5837)](https://www.kaggle.com/code/ryanluoli2/simple-logistic-regression-for-good-score-0-5837)
- **My work**:
    - Implement MLflow, trying out using ML_Ops in main.py
    - Fix the preprocessing issue which seeing the testing dataset before preprocessing the dataset in Baseline work
        - Filling NAs: use median value grouped by product code instead of mean to avoid skewness effect in distribution.  

A slightly better than Baseline in private score