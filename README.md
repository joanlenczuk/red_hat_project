## Predicting Red Hat Business Value


### Overview

This project is inspired by a Kaggle competition available [here](https://www.kaggle.com/c/predicting-red-hat-business-value). 

The aim of the project is to explore data about people and their acitivities and to predict the outcome of specific activities. In this classification problem the main purpose is to identify which activities have the highest potential business value for Red Hat and what characteristics have the most significant influence on the activities' outcomes.

The project was created for self-training purposes.


### Data

The data used in the project contains several files:
- act_train.csv,
- act_test.csv,
- people.csv.

Datasets act_train.csv and act_test.csv contain information about the activities and their characteristics, while people.csv consists of information about people performing those activities. The datasets are available on the website linked above.


### Project description

The project consists of three parts split into three notebooks:
- data_exploration_red_hat.ipynb,
- data_preprocessing_red_hat.ipynb,
- modeling_red_hat.ipynb.

The main purpose of data_exploration_red_hat.ipynb is data exploration. In this notebook data is being imported, cleaned and then explored. The different datasets are being explored separately, as well as jointly. The notebook contains data visualisation of features, the target and the relationship between features and the target. It also captures changes in variables over time and the dependencies between different features.

The aim of data_preprocessing_red_hat.ipynb is to create a preprocessing pipeline. In this script feature selection is being performed based on the correlation coefficients (Cramer's V, point-biserial correlation coefficient) and the frequency score. Missing values are being filled and categorical variables are being encoded using different methods depending on the number of categories.

The main objective of modeling_red_hat.ipynb is to create and evaluate a classification algorithm. Different models (Random Forest, XGBoost, CatBoost) with default parameters are being trained to enable the comparison of performances. Then hyperparameter tuning and threshold adjusting are being performed for the model with the best performance. In the last step feature importance plot and partial dependence plots are being explored to enable a better understanding of the influence of the variables on the outcome.

The project also contains a Python script - red_hat.py. The script consists of all the necessary steps from the three above mentioned notebooks to train the best model for the task, output the results, save model paramaters and the model itself. It includes: importing data, data cleansing, data preprocessing, training the best model and tuning the hyperparameters. 


### Technologies

The project was created using Python 3.7 and JupyterLab.




