# **Wine Quality Prediction**  

This project analyzes and predicts wine quality using machine learning models. The dataset includes different chemical properties of wine, and the goal is to classify wines as good (1) or bad (0).  

## **📌 Steps in the Project**  
1. **Data Preprocessing**  
   - Handle missing values  
   - Encode categorical features  
   - Normalize data  

2. **Exploratory Data Analysis (EDA)**  
   - Visualize distributions and correlations  
   - Generate plots for better insights  

3. **Model Training & Evaluation**  
   - Train models:  
     - Logistic Regression  
     - XGBoost Classifier  
     - Support Vector Machine (SVM)  
   - Evaluate performance using accuracy, precision, and confusion matrix  

## **📂 Files in the Project**  
- `winequalityN.csv` → Dataset  
- `model.py` → Main script for training models  
- `plots/` → Folder containing generated visualizations  

## **🔧 How to Run the Project**  
1. Install required packages:  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
2. Run the project
   '''bash
	python model.py

