# Test04 – Supervised Machine Learning Models (Regression)

## 📌 Problem Statement
The objective of this project is to perform supervised machine learning on a real-world dataset to predict house prices using multiple algorithms and compare their performance.

## 📊 Dataset Description
Dataset: California Housing Dataset  
Rows: 20,640  
Features:  
- longitude  
- latitude  
- housing_median_age  
- total_rooms  
- total_bedrooms  
- population  
- households  
- median_income  
- ocean_proximity (categorical)  

Target Variable: `median_house_value`

## 🧹 Data Cleaning & Preprocessing
Steps performed:

1. **Handling Missing Values**
   - `total_bedrooms` had missing values → filled using median (robust to outliers)

2. **Fixing Data Types**
   - Verified numerical and categorical columns

3. **Outlier Treatment**
   - Dataset inspected using distribution plots (no extreme removal done to preserve data)

4. **Removing Duplicates**
   - Duplicate rows removed

5. **Categorical Encoding**
   - `ocean_proximity` encoded using One-Hot Encoding

6. **Feature Scaling**
   - StandardScaler applied to numerical features

7. **Train-Test Split**
   - 80% training, 20% testing

8. **Skewness Handling**
   - Dataset inspected; no heavy skew transformation required

## 🤖 Algorithms Used
1. Linear Regression  
2. Decision Tree Regressor  
3. Random Forest Regressor  
4. K-Nearest Neighbors  
5. Support Vector Machine (SVR)

## 📈 Evaluation Metrics
Used:
- MSE
- RMSE
- MAE
- R² Score

## 🧪 Results

| Model              | R² Score | RMSE       | MAE       |
|--------------------|----------|------------|-----------|
| Linear Regression  | 0.625    | 70,060     | 50,670    |
| Decision Tree      | 0.636    | 69,078     | 43,550    |
| Random Forest      | 0.817 ✅ | 48,941     | 31,628    |
| KNN                | 0.713    | 61,326     | 40,780    |
| SVM                | -0.043 ❌| 116,917    | 87,042    |

## 🏆 Conclusion / Observations
- Random Forest performed best with highest R² and lowest error values.
- Linear Regression and Decision Tree provided reasonable baseline performance.
- KNN performed moderately well.
- SVM performed poorly without hyperparameter tuning.
- Ensemble models like Random Forest capture complex patterns better in real-world data.
