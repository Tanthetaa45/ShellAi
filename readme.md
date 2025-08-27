# **Shell.ai Hackathon 2025: Fuel Blend Properties Prediction**

## **🚀 Project Overview**

This repository contains the  solution for the **Shell.ai Hackathon 2025: Fuel Blend Properties Prediction Challenge**. The goal of this challenge was to develop a machine learning model to accurately predict 10 key properties of sustainable fuel blends based on their composition and the characteristics of their base components.

An accurate predictive model accelerates the development of new sustainable fuel formulations, enabling a faster transition to a net-zero future without compromising on performance. Our final solution is a sophisticated pipeline that treats each of the 10 blend properties as a unique machine learning problem, resulting in a highly accurate and robust predictive model.

## **📝 Problem Statement**

The challenge, as outlined by Shell, was to predict 10 final blend properties (BlendProperty1 to BlendProperty10) given a dataset containing:

* **Blend Composition**: The volume percentage of 5 base components.  
* **Component Properties**: 10 distinct properties for each of the 5 components.

The evaluation metric for this competition was the **Mean Absolute Percentage Error (MAPE)**.

## **💡 Our Approach: An Iterative Journey to a Hybrid Solution**

We began with a simple, unified approach but quickly discovered that a single model could not effectively capture the unique nature of each of the 10 target properties. This led us on an iterative journey of increasing sophistication, culminating in a powerful, custom pipeline for each target.

### **1\. Exploratory Data Analysis (EDA)**

Before modeling, we conducted a thorough EDA to understand the data's characteristics. This included:

* **Analyzing Distributions**: We checked the statistical properties of each feature.  
* **Skewness Check**: We calculated the skewness for all numeric columns to identify asymmetrical distributions. This step was crucial as it informed our decision to use transformation techniques to normalize the data.

### **2\. Initial Data Preprocessing**

The first step in our pipeline was to ensure the data was clean and robust. This involved:

* **Outlier Handling**: We applied an IQR-based clipping method to remove extreme outliers from all numeric columns, stabilizing our models.  
* **Data Transformation**: Informed by our EDA, we used PowerTransformer to handle skewed data distributions, making them more suitable for modeling.

### **3\. Advanced Feature Engineering**

To provide our models with the richest possible information, we engineered a comprehensive set of new features:

* **Statistical Features**: For each of the 10 base properties, we calculated the mean and standard deviation across the 5 components in a blend. This gave the model a high-level summary of the component profile.  
* **Weighted Features**: We created interaction terms by multiplying each component's property by its corresponding volume fraction (Component\_Property \* Component\_Fraction).  
* **Polynomial Interaction Features**: For each target, we identified the top 10 most important base features and created second-degree interaction terms (feature\_A \* feature\_B). This was the **key breakthrough** that transformed complex, non-linear relationships into patterns that even simple models could solve.

### **4\. The "Model Bake-Off": Finding the Right Tool for Each Job**

Our core insight was that each BlendProperty is a unique problem. We built a "bake-off" pipeline to test three powerful models for each target:

* **Ridge Regression**: A fast and robust linear model.  
* **LightGBM**: A high-performance gradient boosting model.  
* **CatBoost**: Another state-of-the-art gradient boosting model.

### **5\. Advanced Feature Selection & Hyperparameter Tuning**

To maximize performance, we automated two critical steps:

* **Feature Selection**: We experimented with RFECV (Recursive Feature Elimination with Cross-Validation) and a "Top 20" feature selection method using Random Forest importance to find the optimal feature set for each target.  
* **Hyperparameter Tuning**: We used **Optuna**, a powerful Bayesian optimization framework, to efficiently search for the best hyperparameters for our LightGBM and CatBoost models.

This comprehensive, target-specific approach allowed us to significantly reduce the MAPE and create a highly accurate final submission.

## **🛠️ How to Use**

To reproduce the results and generate the final submission.csv file, follow these steps:

1. **Clone the repository:**  
   git clone \[your-repo-link\]  
   cd \[your-repo-name\]

2. **Install the required libraries:**  
   pip install pandas numpy scikit-learn lightgbm catboost optuna

3. Run the final script:  
   The main script (final\_pipeline.py ) is designed to run end-to-end. It will automatically:  
   * Load the train.csv and test.csv files.  
   * Loop through all 10 BlendProperty targets.  
   * For each target, perform the full feature engineering, selection, and model bake-off process.  
   * Identify the best-performing model and use it to predict on the test data.  
   * Combine all predictions into a single submission.csv file.

python final\_pipeline.py

## **🏆 Results**

By treating each target as a unique problem and applying a sophisticated feature engineering and modeling pipeline, we were able to achieve a highly competitive average MAPE score. Our final model for BlendProperty8, for example, achieved a MAPE of **0.94%** after advanced feature engineering, a dramatic improvement from initial scores over 100%.

This project demonstrates the power of a tailored, iterative approach to complex machine learning problems.

## **💻 Technologies Used**

* **Data Manipulation**: Pandas, NumPy  
* **Machine Learning**: Scikit-learn, LightGBM, CatBoost  
* **Hyperparameter Tuning**: Optuna  
* **Visualization**: Matplotlib, Seaborn
