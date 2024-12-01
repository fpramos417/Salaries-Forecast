# University Employee Earnings Forecasting**

This repository contains code for building and evaluating machine learning models to predict employee earnings based on university data. The dataset employed originates from [https://www.kaggle.com/datasets/asaniczka/university-employee-salaries-2011-present](https://www.kaggle.com/datasets/asaniczka/university-employee-salaries-2011-present) (University Employee Salaries, 2011-Present by asanicžka).

**Data Description**

- The dataset encompasses records for university employees, including:
    - `Name` (object)
    - `School` (object)
    - `Job Description` (object)
    - `Department` (object)
    - `Earnings` (float64) - target variable
    - `Year` (int64)

**Data Preprocessing**

1. **Missing Values:**
   - The code identifies and removes rows with missing values in any column except `Year`. You might consider additional strategies like imputation (filling in missing values) depending on the data distribution and business context.

2. **Feature Engineering:**
   - `Name` is dropped as it's unlikely to be relevant for predicting earnings.
   - Categorical features (`School`, `Job Description`, `Department`) are encoded using One-Hot Encoding to convert them into numerical representations suitable for machine learning algorithms.

3. **Data Splitting:**
   - The code employs K-Fold cross-validation instead of a traditional train-test split. K-Fold ensures that each data point has a chance to be part of the test set, leading to more robust model evaluation.

4. **Data Standardization:**
   - Standard scaling is applied using `StandardScaler` to normalize numerical features (Year) and prevent them from biasing the model towards features with larger magnitudes.

**Machine Learning Models**

The code currently implements the following models:

- **Linear Regression (Ridge):** A regularized linear model that can handle correlated features and reduce overfitting.
- **Decision Tree: A non-parametric model that can capture complex relationships between features and the target variable.

**Evaluation Metrics**

- **Root Mean Squared Error (RMSE):** Measures the average difference between predicted and actual earnings. Lower RMSE indicates better prediction accuracy.
- **R-squared (R²):** Assesses the proportion of variance in earnings explained by the model. Higher R² signifies a stronger relationship between features and earnings.

**Evaluation

Based on the evaluation metrics (RMSE and R²), the Decision Tree model outperforms the Ridge Regression model. Therefore, the Decision Tree is selected as the final model for predicting employee earnings.
**Code Structure**

1. **Imports:** Necessary libraries like pandas, NumPy, scikit-learn modules for data manipulation, modeling, and evaluation are imported.
2. **Data Loading:** The `higher_ed_sal.csv` dataset is read using `pd.read_csv`.
3. **Data Exploration:**
   - Basic data information is presented using `data.shape`, `data.info()`, and data visualization techniques (consider using libraries like seaborn or matplotlib).
4. **Data Cleaning:** Missing values are handled (currently by removal). Additional steps might involve imputation or handling outliers depending on data analysis.
5. **Feature Engineering:**
   - `Name` is dropped.
   - Categorical features are encoded using one-hot encoding.
   - You might also consider feature engineering techniques like creating new features based on domain knowledge.
6. **Data Splitting:** K-Fold cross-validation is set up.
7. **Model Pipeline Definition:**
   - A reusable pipeline is created using `Pipeline` to streamline data preprocessing, scaling, and model fitting.
   - The pipeline encompasses:
     - `OneHotEncoder` (with `handle_unknown='ignore'`) to handle potential unknown categories in categorical features.
     - `ColumnTransformer` to specify which columns to encode.
     - `StandardScaler` to scale numerical features.
     - The chosen model (e.g., `Ridge()`, `DecisionTreeRegressor()`)
8. **Model Evaluation:**
   - The `evaluate_model` function is defined to perform K-Fold cross-validation, train the model on each fold, predict on the test fold, and calculate average RMSE and R².
9. **Model Training and Evaluation:**
   - Each model in the `models` dictionary (currently containing only Ridge Regression) is trained and evaluated using K-Fold cross-validation.
   - The results (average RMSE and R²) are printed for each model to compare their performance. Based on the provided data, the Decision Tree is currently performing better.
10. **Prediction on New Data:**
   - The code demonstrates how to make predictions on a new dataset using the trained model (currently the Decision Tree).

**Further Considerations**

- Experiment with different machine learning models (e.g., Random Forest, Gradient Boosting) to identify the best fit for your dataset and business needs.
- Consider hyperparameter tuning to optimize the performance of each model. You might use
