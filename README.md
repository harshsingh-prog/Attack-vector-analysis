# Attack Vector Analysis

## Project Description

Attack Vector Analysis is a machine learning project aimed at identifying and analyzing potential attack vectors in software systems. By leveraging various machine learning techniques, this project aims to enhance the detection and prevention of security threats.

## Features

- **Data Preprocessing**: Cleaning and preparing the dataset for analysis.
- **Machine Learning Models**: Implementing and training models to identify attack vectors.
- **Model Evaluation**: Assessing the performance of the models using various metrics.
- **Data Visualization**: Presenting insights through visualizations to better understand the data and model performance.

## Technologies Used

- **Programming Language**: Python
- **Data Analysis**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Standard Scaler, KNeighborsClassifier, RandomForestClassifier
- **Hyperparameter Tuning**: Optuna
- **Data Visualization**: Matplotlib, Seaborn

## Usage

1. Prepare your dataset (e.g., `Phishing_Legitimate_full.csv`) and place it in the project directory.
2. Run the data preprocessing script to clean and prepare the data:
    ```bash
    python preprocess.py
    ```
3. Train the machine learning model:
    ```bash
    python train_model.py
    ```
4. Evaluate the model performance:
    ```bash
    python evaluate_model.py
    ```
5. Generate visualizations to understand the data and model performance:
    ```bash
    python visualize.py
    ```

## Results

- The machine learning models effectively identify attack vectors, achieving high accuracy scores.
- Data visualizations provide clear insights into attack vector patterns and model performance, aiding in informed decision-making for enhancing security measures.

## Code Overview

The project is structured as follows:

- **Data Preprocessing**:
    - **Loading Dataset**: The dataset is loaded using Pandas.
    - **Cleaning Data**: Irrelevant columns are dropped, and missing values are handled.
    - **Encoding**: Categorical features are encoded to numerical values for model training.
- **Model Training**:
    - **Splitting Data**: The dataset is split into training and testing sets.
    - **Scaling**: Features are scaled using StandardScaler.
    - **Training**: KNeighborsClassifier and RandomForestClassifier are used to train the models.
- **Hyperparameter Tuning**:
    - **Optuna**: Used for tuning the hyperparameters of the RandomForestClassifier.
- **Model Evaluation**:
    - **Metrics**: Accuracy, precision, recall, and F1-score are calculated.
- **Data Visualization**:
    - **Distribution Plots**: Histograms, count plots, box plots, scatter plots, and correlation matrices are used to visualize the data.


    git checkout -b feature-branch
    ```
3. Make your changes and commit them:
    ```bash
