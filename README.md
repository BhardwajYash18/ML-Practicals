# Machine Learning Practicals

This repository contains my **Machine Learning Laboratory practicals**, including Jupyter Notebooks, datasets, implementations, visualizations, and related practical work completed as part of the Machine Learning course.

The repository will be updated as new practicals are completed.

---

## 👨‍💻 Student Details

| Detail | Information |
|---|---|
| **Name** | Yash Bhardwaj |
| **Section** | A |
| **Roll No.** | 09 |
| **Batch** | A-A1 |
| **Course** | Machine Learning |

---

# 📚 Practicals

| Lab No. | Practical | Dataset / Topic | Status |
|---|---|---|---|
| **Lab 1** | Data Preprocessing | Titanic Dataset | ✅ Completed |
| **Lab 2** | Regression Models | USA Housing Dataset | ✅ Completed |
| **Lab 3** | — | — | ⏳ Pending |
| **Lab 4** | — | — | ⏳ Pending |
| **Lab 5** | — | — | ⏳ Pending |

---

# 🧪 Lab 1 — Data Preprocessing

## Aim

To study and apply **Data Preprocessing techniques** on the given dataset and prepare the **Titanic dataset** for training with a machine learning algorithm.

## Dataset

**Titanic Dataset**

The dataset contains passenger-related information including:

- Passenger ID
- Survival Status
- Passenger Class
- Name
- Sex
- Age
- Number of Siblings / Spouses
- Number of Parents / Children
- Ticket
- Fare
- Cabin
- Port of Embarkation

The `Survived` column is used as the target variable.

The original dataset contains **891 records and 12 attributes**.

## Data Preprocessing Steps

The following preprocessing steps are performed:

1. Data Inspection
2. Missing Value Handling
3. Removing Unnecessary Columns
4. Categorical Data Encoding
5. Duplicate Data Checking
6. Outlier Detection and Removal
7. Exploratory Data Analysis and Visualization
8. Feature Scaling
9. Feature and Target Separation
10. Train-Test Split

## Missing Value Handling

Missing values are identified using `isnull().sum()`.

The dataset initially contains:

- **Age:** 177 missing values
- **Cabin:** 687 missing values
- **Embarked:** 2 missing values

The missing `Age` values are handled using the median, while missing `Embarked` values are handled using the mode.

The `Cabin` column is removed because of its large number of missing values.

## Categorical Data Encoding

Categorical variables are converted into numerical form.

The `Embarked` feature is mapped as:

```text
S → 0
C → 1
Q → 2
