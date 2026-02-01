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
- ocean_proximity  

Target Variable: `median_house_value`

## 🧹 Data Cleaning & Preprocessing
1. Missing values handled (median used for total_bedrooms)  
2. Duplicate rows removed  
3. Categorical variable encoded using One-Hot Encoding  
4. Feature scaling using StandardScaler  
5. Train-test split (80% train, 20% test)  
6. Checked skewness and data distributions  

## 🤖 Algorithms Used
1. Linear Regression  
2. Decision Tree  
3. Random Forest  
4. K-Nearest Neighbors (KNN)  
5. Support Vector Machine (SVM)  

## 📈 Evaluation Metrics
- MSE  
- RMSE  
- MAE  
- R² Score  

## 🧪 Results

| Model             | R² Score | RMSE     | MAE     |
|-------------------|----------|----------|---------|
| Linear Regression | 0.625    | 70060    | 50670   |
| Decision Tree     | 0.636    | 69078    | 43550   |
| Random Forest     | 0.817 ✅ | 48941    | 31628   |
| KNN               | 0.713    | 61326    | 40780   |
| SVM               | -0.043 ❌| 116917   | 87042   |

## 🏆 Conclusion
Random Forest performed the best with the highest R² score and lowest error values.  
Linear Regression and Decision Tree gave reasonable baseline performance.  
KNN performed moderately well.  
SVM performed poorly without hyperparameter tuning.  
Ensemble models like Random Forest captured complex patterns better for this dataset.
