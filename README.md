# Machine Learning Practicals

A collection of Machine Learning laboratory practicals implemented in Python using Jupyter Notebooks. This repository covers data preprocessing, regression, concept learning, decision trees, K-Nearest Neighbors, Artificial Neural Networks, and Naive Bayes classification.

## Student Information

| Field | Details |
|---|---|
| Name | Yash Bhardwaj |
| Section | A |
| Roll Number | 09 |
| Batch | A-A1 |
| Subject | Machine Learning |

## Table of Contents

1. [Overview](#overview)
2. [Practical 1: Data Preprocessing](#practical-1-data-preprocessing)
3. [Practical 2: Linear Regression Models](#practical-2-linear-regression-models)
4. [Practical 3: Find-S and Candidate Elimination](#practical-3-find-s-and-candidate-elimination)
5. [Practical 4: Decision Tree Classification](#practical-4-decision-tree-classification)
6. [Practical 5: K-Nearest Neighbors](#practical-5-k-nearest-neighbors-knn)
7. [Practical 6: Artificial Neural Network](#practical-6-artificial-neural-network-ann)
8. [Practical 7: Naive Bayes Classification](#practical-7-naive-bayes-classification)
9. [Technologies Used](#technologies-used)
10. [Repository Structure](#repository-structure)
11. [Installation and Setup](#installation-and-setup)
12. [How to Run](#how-to-run)
13. [Learning Outcomes](#learning-outcomes)

---

## Overview

This repository contains seven Machine Learning practicals covering fundamental algorithms, data analysis techniques, model training, and evaluation.

| Practical | Topic | Dataset / Resource | Algorithm or Technique |
|---|---|---|---|
| 1 | Data Preprocessing | Titanic dataset | Data cleaning and transformation |
| 2 | Linear Regression Models | USA Housing dataset | Simple, Multiple, Ridge and Lasso Regression |
| 3 | Find-S and Candidate Elimination | EnjoySport dataset | Concept learning |
| 4 | Decision Tree Classification | Wine Quality Red dataset | Decision Tree Classifier |
| 5 | K-Nearest Neighbors | Diabetes dataset | KNN Classification and Regression |
| 6 | Artificial Neural Network | Seeds dataset | Multilayer Perceptron (MLP) |
| 7 | Naive Bayes Classification | Raisin Dataset | Gaussian Naive Bayes |

---

## Practical 1: Data Preprocessing

**Objective:** To understand and perform data preprocessing operations required to prepare raw data for Machine Learning.

**Dataset:** Titanic dataset (`titanic.csv`)

### Topics Covered

- Loading datasets using Pandas.
- Inspecting dataset structure and dimensions.
- Identifying missing values.
- Handling missing data.
- Data cleaning and transformation.
- Preparing data for further analysis and Machine Learning.

### Tools and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn

**Notebook:** `YashBhardwaj_ML-1.ipynb`

**Folder:** [`Lab-1_Data-Preprocessing`](./Lab-1_Data-Preprocessing/)

---

## Practical 2: Linear Regression Models

**Objective:** To implement and compare different regression techniques for predicting continuous numerical values.

**Dataset:** USA Housing dataset (`USA_Housing.csv`)

### Models Implemented

1. Simple Linear Regression
2. Multiple Linear Regression
3. Ridge Regression
4. Lasso Regression

### Topics Covered

- Loading and exploring the dataset.
- Selecting input features and the target variable.
- Splitting data into training and testing sets.
- Training regression models.
- Generating predictions.
- Comparing regression models using evaluation metrics.

### Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

**Notebook:** `YashBhardwaj_Practical_2_MLR.ipynb`

**Folder:** [`Lab-2_Linear-Regression-Models`](./Lab-2_Linear-Regression-Models/)

---

## Practical 3: Find-S and Candidate Elimination

**Objective:** To implement concept learning algorithms and understand how hypotheses are generalized or specialized based on training examples.

**Dataset:** EnjoySport dataset (`enjoy.csv`)

### Algorithms Implemented

- Find-S Algorithm
- Candidate Elimination Algorithm

### Topics Covered

- Concept learning.
- Hypothesis representation.
- Positive and negative training examples.
- General and specific hypotheses.
- Finding the most specific consistent hypothesis.
- Maintaining the version space.
- Updating hypothesis boundaries using training data.

### Tools and Libraries

- Python
- Pandas

**Notebook:** `A-A1-09-YashBhardwaj_ML-3.ipynb`

**Folder:** [`Lab-3_Find-S_And_CandidateElimination_Algorithm`](./Lab-3_Find-S_And_CandidateElimination_Algorithm/)

---

## Practical 4: Decision Tree Classification

**Objective:** To implement a Decision Tree Classifier on the Wine Quality Red dataset and analyze its classification performance.

**Dataset:** Wine Quality Red dataset (`winequality-red.csv`)

The dataset contains physicochemical measurements of red wine. The `quality` column is used as the target variable.

### Topics Covered

- Dataset loading and inspection.
- Statistical summaries and data validation.
- Missing-value and duplicate checks.
- Exploratory Data Analysis (EDA).
- Histograms and box plots.
- Outlier analysis using the Interquartile Range (IQR).
- Univariate, bivariate and multivariate analysis.
- Feature and target separation.
- Training and testing data splitting.
- Decision Tree Classification using the entropy criterion.
- Model training and prediction.
- Classification evaluation metrics.
- Confusion matrix visualization.
- Decision tree depth and leaf analysis.
- Decision tree visualization.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

**Folder:** [`Lab-4_Decision_Tree_Classification`](./Lab-4_Decision_Tree_Classification/)

---

## Practical 5: K-Nearest Neighbors (KNN)

**Objective:** To implement K-Nearest Neighbors for both classification and regression using the Diabetes dataset.

**Dataset:** `diabetes (6).csv`

### Part A: KNN Classification

The classification model predicts the `Outcome` variable, representing the diabetes classification.

### Part B: KNN Regression

The regression model predicts the `BMI` variable using the remaining eligible features.

### Topics Covered

- Dataset loading and inspection.
- Data cleaning and preprocessing.
- Handling invalid zero values where appropriate.
- Missing-value imputation.
- Feature and target selection.
- Feature scaling using `StandardScaler`.
- Training and testing data splitting.
- Implementing KNN classification.
- Implementing KNN regression.
- Comparing different values of K.
- Cross-validation and hyperparameter selection.
- Evaluating classification and regression models.

### Classification Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Regression Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

### Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

**Folder:** [`Lab-5_KNN_Algorithm.`](./Lab-5_KNN_Algorithm./)

---

## Practical 6: Artificial Neural Network (ANN)

**Objective:** To implement a Multilayer Perceptron (MLP) classifier using TensorFlow and Keras for multiclass classification.

**Dataset:** Seeds dataset (`seeds_new.csv`)

The model classifies samples into three seed varieties using their numerical features.

### Model Architecture

The neural network consists of:

- Input layer corresponding to the dataset features.
- First hidden Dense layer with 16 neurons and ReLU activation.
- Second hidden Dense layer with 8 neurons and ReLU activation.
- Output Dense layer with 3 neurons and Softmax activation.

### Topics Covered

- Dataset loading and inspection.
- Feature and target separation.
- Encoding target classes.
- Feature standardization using `StandardScaler`.
- Training and testing data splitting.
- Building a neural network using TensorFlow and Keras.
- Compiling the model with the Adam optimizer.
- Using sparse categorical cross-entropy as the loss function.
- Training the network.
- Early stopping to help prevent overfitting.
- Predicting multiclass outputs.
- Evaluating classification performance.

### Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras

**Folder:** [`Lab-6_ANN_Classifier`](./Lab-6_ANN_Classifier/)

---

## Practical 7: Naive Bayes Classification

**Objective:** To implement Gaussian Naive Bayes for classifying raisin varieties based on their numerical measurements.

**Dataset:** Raisin Dataset (`Raisin_Dataset.xlsx`)

The dataset contains 900 samples and eight columns, including seven numerical features and the categorical target variable `Class`.

The two target classes are:

- `Kecimen`
- `Besni`

### Algorithm Used

**Gaussian Naive Bayes (`GaussianNB`)**

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It models numerical features using Gaussian distributions within each class and assumes conditional independence between features given the class.

### Topics Covered

- Loading an Excel dataset using Pandas.
- Dataset inspection and statistical summaries.
- Checking missing values and duplicate records.
- Exploratory Data Analysis (EDA).
- Class distribution visualization.
- Histograms and box plots.
- Kernel Density Estimation (KDE) plots.
- Correlation heatmap.
- Scatter plots for feature relationships.
- Encoding target labels using `LabelEncoder`.
- Separating features and target variables.
- Splitting data into training and testing sets.
- Training a Gaussian Naive Bayes classifier.
- Generating predictions.
- Evaluating classification performance.
- Comparing training and testing accuracy.
- Experimenting with different `var_smoothing` values.
- Predicting the class of new raisin samples.

### Evaluation Metrics

- Accuracy
- Weighted precision
- Weighted recall
- F1-score
- Classification report
- Confusion matrix

### Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- `openpyxl` for reading Excel files

**Notebook:** `A-A1-09_YashBhardwaj_ML_Practical-7.ipynb`

**Dataset:** `Raisin_Dataset.xlsx`

**Folder:** [`Lab-7_Naive_Bayes_Classification`](./Lab-7_Naive_Bayes_Classification/)

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Jupyter Notebook | Interactive development and experimentation |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Seaborn | Statistical data visualization |
| Scikit-learn | Machine Learning algorithms, preprocessing and evaluation |
| TensorFlow | Neural network implementation |
| Keras | Building and training neural networks |
| openpyxl | Reading Excel datasets |

---

## Repository Structure

```text
ML-Practicals/
│
├── Lab-1_Data-Preprocessing/
│   └── Titanic dataset and notebook
│
├── Lab-2_Linear-Regression-Models/
│   └── USA Housing dataset and notebook
│
├── Lab-3_Find-S_And_CandidateElimination_Algorithm/
│   └── EnjoySport dataset and notebook
│
├── Lab-4_Decision_Tree_Classification/
│   └── Wine Quality dataset and notebook
│
├── Lab-5_KNN_Algorithm./
│   └── Diabetes dataset and notebook
│
├── Lab-6_ANN_Classifier/
│   └── Seeds dataset and notebook
│
├── Lab-7_Naive_Bayes_Classification/
│   ├── A-A1-09_YashBhardwaj_ML_Practical-7.ipynb
│   └── Raisin_Dataset.xlsx
│
└── README.md
```

*Note: The notebook and dataset descriptions above identify the resources used in each practical. Actual filenames may differ in some folders; check each folder for its exact contents.*

---

## Installation and Setup

### 1. Install Python

Install Python 3 from the official website:

https://www.python.org/downloads/

### 2. Clone the Repository

```bash
git clone https://github.com/BhardwajYash18/ML-Practicals.git
```

### 3. Navigate to the Project Directory

```bash
cd ML-Practicals
```

### 4. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

### 5. Install the Required Libraries

Install the common dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter openpyxl
```

For Practical 6, install TensorFlow:

```bash
pip install tensorflow
```

---

## How to Run

1. Clone the repository.
2. Install the required dependencies.
3. Navigate to the folder of the practical you want to execute.
4. Ensure the corresponding dataset is available at the file path specified in the notebook.
5. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

6. Open the relevant `.ipynb` file.
7. Run the cells sequentially, from top to bottom.

**Important:** Some notebooks use relative dataset paths. Keep each dataset in the expected location or update the file path in the notebook before running it.

---

## Learning Outcomes

By completing these practicals, students gain experience with:

- Data inspection, cleaning and preprocessing.
- Exploratory Data Analysis and visualization.
- Simple and multiple linear regression.
- Regularization using Ridge and Lasso.
- Concept learning using Find-S and Candidate Elimination.
- Decision Tree classification and entropy.
- KNN classification and regression.
- Feature scaling and hyperparameter selection.
- Artificial Neural Networks and multilayer perceptrons.
- Gaussian Naive Bayes classification.
- Model training, testing and prediction.
- Evaluating Machine Learning models using suitable metrics.
- Implementing Machine Learning algorithms with Python and Scikit-learn.

---

## Author

**Yash Bhardwaj**

- GitHub: [BhardwajYash18](https://github.com/BhardwajYash18)

---

*This repository is maintained for academic learning and Machine Learning laboratory practice.*
