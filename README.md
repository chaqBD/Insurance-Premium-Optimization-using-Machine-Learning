Insurance Premium Optimization using Machine Learning

This project aims to build an optimized system for estimating insurance premiums using machine learning techniques. Traditional insurance pricing models, such as Generalized Linear Models (GLMs), often fail to capture the nonlinearities in insurance data, leading to suboptimal premium pricing. This project employs machine learning algorithms to improve accuracy, transparency, and adaptability in insurance premium estimation.

Project Overview

This repository contains the implementation of several machine learning models for insurance premium optimization. The goal is to develop a robust system for estimating insurance premiums that outperforms traditional methods in terms of accuracy and adaptability. The models have been rigorously validated, and hyperparameter tuning was performed to achieve optimal results.

Key Features

Machine Learning Models Used: Linear Regression, Support Vector Machine (SVM), Random Forest, XGBoost, and Deep Feedforward Neural Networks (DFNN).

Data Used: The dataset contains 1338 observations with features such as age, Body Mass Index (BMI), number of children, smoking status, and region.

Feature Engineering: Includes label encoding for categorical variables, normalization of numerical features, and feature selection using Mutual Information and Recursive Feature Elimination (RFE).

Optimization Techniques: Extensive hyperparameter tuning and early stopping were applied to improve model performance.

Performance Metrics: The models were evaluated using R², Root Mean Square Error (RMSE), and Mean Absolute Error (MAE). XGBoost showed the best performance, achieving the highest R² and lowest RMSE.

Dataset

The dataset used in this project is publicly available and was sourced from Kaggle. It contains the following features:

Age: Age of the individual (numerical).

BMI: Body Mass Index (numerical).

Children: Number of dependents (numerical).

Smoker: Whether the individual is a smoker (categorical).

Region: Geographic region of residence (categorical).

Expenses: Medical expenses incurred (numerical, target variable).

The dataset was preprocessed to handle missing values, encode categorical variables, and normalize numerical features to ensure compatibility with machine learning algorithms.

Methodology

Data Preprocessing:

Missing Values Handling: Checked for missing values (none found).

Categorical Encoding: Categorical features (sex, smoker, region) were label encoded.

Normalization: Numerical features (age, BMI, expenses) were normalized to bring them to a similar scale.

Exploratory Data Analysis (EDA): Visualizations such as histograms, count plots, and heatmaps were used to understand data distribution, correlations, and relationships between variables.

Feature Selection:

Mutual Information: Quantified the relevance of features in predicting insurance expenses.

Recursive Feature Elimination (RFE): Iteratively selected the most important features using linear regression as a base estimator.

Model Development:

Five models were implemented: Linear Regression, SVM, Random Forest, XGBoost, and Deep Feedforward Neural Networks.

Hyperparameter Tuning: Applied Grid Search and Random Search to find optimal parameters for each model.

Model Evaluation:

Models were evaluated based on R², RMSE, and MAE.

XGBoost achieved the best results, indicating superior performance in handling complex nonlinear relationships.

Results

XGBoost showed the highest R² value and the lowest RMSE, making it the most suitable model for insurance premium prediction in this study.

Linear Regression was used as a baseline and demonstrated that more complex models are needed to capture the nonlinearities in the data.

Deep Feedforward Neural Network (DFNN) provided competitive results but was outperformed by XGBoost in terms of accuracy.

Installation and Usage

To run this project, clone the repository and install the required Python packages:

$ git clone https://github.com/yourusername/insurance-optimization.git
$ cd insurance-optimization
$ pip install -r requirements.txt

After installing the dependencies, run the Jupyter Notebook to execute the complete workflow:

$ jupyter notebook insurance_optimization.ipynb

Dependencies

Python 3.7+

Pandas

NumPy

Scikit-Learn

XGBoost

TensorFlow

Matplotlib

Seaborn

Install the dependencies using the requirements.txt file provided.

Directory Structure

data/ - Contains the dataset.

notebooks/ - Jupyter notebooks for data analysis, model development, and evaluation.

models/ - Saved model files after training.

src/ - Source code for preprocessing, training, and evaluation.

Future Work

Deployment: Deploy the best-performing model using Flask or FastAPI to create a REST API for real-time insurance premium estimation.

Explainability: Integrate explainable AI techniques to make the models more transparent for stakeholders.

Further Optimization: Experiment with additional optimization algorithms like LightGBM and CatBoost.

Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements or new features.

License

This project is licensed under the MIT License. See LICENSE for more information.

Contact

For any questions or collaboration opportunities, please contact: TEWOGBADE SHAKIR, pingcommercial@gmail.com 

