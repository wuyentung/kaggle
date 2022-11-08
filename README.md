# kaggle with technologies
open competition experiences in kaggle, with self-learned technology implementations.
## Time-Series Prediction Review
Forecast a whole year numbers of selling in six countries and four books.
- kaggle competition: Tabular Playground Series - Sep 2022
- Baseline: [[TPS-SEP-22] EDA and Linear Regression Baseline](https://www.kaggle.com/code/cabaxiom/tps-sep-22-eda-and-linear-regression-baseline).
- **My work**: linear regression with feature engineering by week.  

It seems that the baseline version made a better prediction with feature engineering in day of year.  
## MLOps
Predict individual product failures (in probability) of new codes with their individual lab test results.
- kaggle competition: Tabular Playground Series - Aug 2022
- Baseline: [Simple Logistic Regression for Good Score (0.5837)](https://www.kaggle.com/code/ryanluoli2/simple-logistic-regression-for-good-score-0-5837)
- **My work**:
    - Implement MLflow with cross-validation in main.py
    - Fix the preprocessing issue which seeing the testing dataset before preprocessing the dataset in Baseline work
        - Filling NAs: use median value grouped by product code instead of mean to avoid skewness effect in distribution.  

A slightly better than Baseline in private score  
## Pyspark
Predict Median House Value, the dependent variable refers to the median house value per block group, among continuous independent variables.
- Dataset from [housing_data](https://www.kaggle.com/datasets/fatmakursun/hausing-data) in kaggle.
- This work is inspired by [Pyspark ML tutorial for beginners](https://www.kaggle.com/code/fatmakursun/pyspark-ml-tutorial-for-beginners/notebook).