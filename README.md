# Machine Learning Practicals

This repository contains my **Machine Learning Laboratory practicals**, including Jupyter Notebooks, datasets, implementations, visualizations, and related practical work completed as part of the Machine Learning course.

The repository will be updated as new practicals are completed.

---

## 👨‍💻 Student Details

| Detail       | Information       |
|--------------|--------------------|
| **Name**     | Yash Bhardwaj      |
| **Section**  | A                  |
| **Roll No.** | 09                 |
| **Batch**    | A-A1               |
| **Course**   | Machine Learning   |

---

# 📚 Practicals

| Lab No.   | Practical                                   | Dataset / Topic      | Status       |
|-----------|----------------------------------------------|-----------------------|--------------|
| **Lab 1** | Data Preprocessing                           | Titanic Dataset       | ✅ Completed |
| **Lab 2** | Regression Models (SLR, MLR, Ridge, Lasso)   | USA Housing Dataset   | ✅ Completed |
| **Lab 3** | Find-S & Candidate Elimination Algorithm     | EnjoySport Dataset    | ✅ Completed |
| **Lab 4** | —                                             | —                      | ⏳ Pending   |
| **Lab 5** | —                                             | —                      | ⏳ Pending   |

---

# 🧪 Lab 1 — Data Preprocessing

**Folder:** [`Lab-1_Data-Preprocessing`](./Lab-1_Data-Preprocessing) &nbsp;|&nbsp; **Notebook:** `YashBhardwaj_ML-1.ipynb` &nbsp;|&nbsp; **Data:** `titanic.csv`

## Aim

To study and apply **data preprocessing techniques** on the given dataset and prepare the **Titanic dataset** for training with a machine learning algorithm.

## Dataset

**Titanic Dataset** — the dataset contains passenger-related information including:

- Passenger ID
- Survival Status (`Survived`)
- Passenger Class (`Pclass`)
- Name
- Sex
- Age
- Number of Siblings / Spouses (`SibSp`)
- Number of Parents / Children (`Parch`)
- Ticket
- Fare
- Cabin
- Port of Embarkation (`Embarked`)

The `Survived` column is used as the **target variable**. The original dataset contains **891 records and 12 attributes**.

## Data Preprocessing Steps

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

### 1. Missing Value Handling

Missing values are identified using `isnull().sum()`. The dataset initially contains:

| Column       | Missing Values |
|--------------|-----------------|
| **Age**      | 177             |
| **Cabin**    | 687             |
| **Embarked** | 2               |

- Missing `Age` values are filled using the **median**.
- Missing `Embarked` values are filled using the **mode**.
- The `Cabin` column is **dropped** entirely due to its large proportion of missing values.

### 2. Categorical Data Encoding

Categorical variables are converted into numerical form so they can be used by the model.

`Embarked` is manually mapped as:

```text
S → 0
C → 1
Q → 2
```

`Sex` is encoded using scikit-learn's `LabelEncoder` (`male → 0`, `female → 1`).

### 3. Duplicate Data Checking

Duplicate rows are checked using `df.duplicated().sum()`. No duplicate records are found in the dataset.

### 4. Outlier Detection and Removal

- Boxplots are used to visualize outliers in the **Age** and **Fare** columns.
- Outliers in **Fare** are removed using the **IQR (Interquartile Range) method**:
  `lower = Q1 − 1.5×IQR`, `upper = Q3 + 1.5×IQR`.

### 5. Exploratory Data Analysis and Visualization

- Pie chart of the `Sex` distribution
- Histogram (with KDE) of `Age`
- Count plot of `Survived`
- Count plot of `Sex` vs. `Survived`
- Scatter plot of `Age` vs. `Fare`
- Correlation heatmap of numeric features
- Pairplot across all numeric features

### 6. Feature Scaling

`Age` and `Fare` are standardized using scikit-learn's `StandardScaler` so both features are on a comparable scale.

### 7. Feature and Target Separation

```python
X = df.drop("Survived", axis=1)
y = df["Survived"]
```

### 8. Train-Test Split

The data is split **80% train / 20% test** using `train_test_split(..., test_size=0.20, random_state=42)`.

---

# 🧪 Lab 2 — Regression Models (Simple, Multiple, Ridge & Lasso)

**Folder:** [`Lab-2_Linear-Regression-Models`](./Lab-2_Linear-Regression-Models) &nbsp;|&nbsp; **Notebook:** `YashBhardwaj_Practical_2_MLR.ipynb` &nbsp;|&nbsp; **Data:** `USA_Housing.csv`

## Aim

To study and implement **Simple Linear Regression**, **Multiple Linear Regression**, and **regularized regression (Ridge & Lasso)** with hyperparameter tuning, in order to predict house prices from the USA Housing dataset.

## Dataset

**USA Housing Dataset** — **5000 records and 7 attributes**:

- Avg. Area Income
- Avg. Area House Age
- Avg. Area Number of Rooms
- Avg. Area Number of Bedrooms
- Area Population
- Price *(target variable)*
- Address *(dropped — non-numeric, not predictive)*

## Workflow

1. **Data Loading & Inspection** — `head()`, `tail()`, `shape`, `describe()`
2. **Column Cleanup** — the `Address` column is dropped as it holds no predictive value
3. **Missing Value & Duplicate Checks** — `isnull().sum()` and `duplicated().sum()`
4. **Outlier Detection & Removal** — boxplots + IQR method applied to `Avg. Area Income` and `Price`
5. **Exploratory Data Analysis** — boxplots, correlation heatmap, and distribution plots (histogram + KDE) for `Price` and `Avg. Area Income`
6. **Simple Linear Regression (SLR)** — `Avg. Area Income` → `Price`
7. **Multiple Linear Regression (MLR)** — all remaining numeric features → `Price`
8. **Ridge Regression** — baseline model, then tuned via `GridSearchCV`
9. **Lasso Regression** — tuned via `GridSearchCV`
10. **Custom Prediction** — takes user input and predicts house price interactively

## Results

### Simple Linear Regression (Avg. Area Income → Price)

| Metric | Value |
|--------|-------|
| MAE    | 216,087.71 |
| MSE    | 70,880,962,067.45 |
| RMSE   | 464.85 |
| R² Score | 0.398 |

### Multiple Linear Regression (all features → Price)

| Metric | Value |
|--------|-------|
| MAE    | 78,456.94 |
| MSE    | 9,643,840,322.41 |
| RMSE   | 98,203.06 |
| R² Score | 0.918 |

### Ridge Regression (tuned via GridSearchCV, `cv=5`)

- Parameter grid: `alpha = [0.001, 0.01, 1, 10, 100]`
- **Best alpha:** `1` (best CV R² ≈ 0.910)

| Metric | Value |
|--------|-------|
| MAE    | 78,456.01 |
| MSE    | 9,643,550,061.42 |
| RMSE   | 98,201.58 |
| R² Score | 0.918 |

### Lasso Regression (tuned via GridSearchCV, `cv=5`)

- Parameter grid: `alpha = [0.001, 0.01, 0.1, 1, 10]`, `max_iter=5000`
- **Best alpha:** `0.001`
- **R² Score:** 0.918

**Takeaway:** Multiple Linear Regression dramatically outperforms Simple Linear Regression (R² of **0.92** vs. **0.40**), since house price depends on several features beyond income alone. Ridge and Lasso regularization achieve virtually the same accuracy as plain MLR here, indicating minimal overfitting/multicollinearity in this dataset.

---

# 🧪 Lab 3 — Find-S & Candidate Elimination Algorithm

**Folder:** [`Lab-3_Find-S_And_CandidateElimination_Algorithm`](./Lab-3_Find-S_And_CandidateElimination_Algorithm) &nbsp;|&nbsp; **Notebook:** `A-A1-09-YashBhardwaj_ML-3.ipynb` &nbsp;|&nbsp; **Data:** `enjoy.csv`

## Aim

To implement the **Find-S algorithm** and the **Candidate Elimination algorithm** for concept learning — deriving the most specific and the version-space (specific + general boundary) hypotheses consistent with a set of training examples.

## Dataset

**EnjoySport Dataset** (`enjoy.csv`) — the classic concept-learning dataset with **10 records and 7 attributes**:

- Sky
- AirTemp
- Humidity
- Wind
- Water
- Forecast
- EnjoySport *(target variable — `Yes`/`No`)*

The dataset has **no missing values and no duplicate rows** (verified with `isnull().sum()` and `duplicated().sum()`).

## Workflow

1. **Data Loading & Inspection** — `head()`, `tail()`, `info()`, `describe()`, `shape`
2. **Missing Value & Duplicate Checks**
3. **Feature/Target Split** — `X = df.iloc[:, :-1]`, `y = df.iloc[:, -1]`
4. **Find-S Algorithm** — starts with the most specific hypothesis and generalizes it using only the **positive** (`Yes`) training examples
5. **Candidate Elimination Algorithm** — maintains both a **specific boundary (S)** and a **general boundary (G)**, updating S on positive examples and G on negative examples to converge on the version space

## Results

### Find-S Algorithm

Starting from `None`, the hypothesis is generalized step by step over each positive example. The **final maximally-specific hypothesis**:

```text
['Sunny', 'Warm', '?', '?', '?', '?']
```

This means: EnjoySport = Yes whenever `Sky = Sunny` and `AirTemp = Warm`, regardless of Humidity, Wind, Water, or Forecast.

### Candidate Elimination Algorithm

Run over all 10 examples, tracking S and G after each one.

**Final Specific Boundary (S):**

```text
['Sunny', 'Warm', '?', '?', '?', '?']
```

**Final General Boundary (G):** contains `['Sunny', ?, ?, ?, ?, ?]` and `['?', 'Warm', ?, ?, ?, ?]` as the two maximally-general consistent hypotheses (matching S).

> **Note:** the notebook's `general` list accumulates an entry for *every* attribute mismatched on a negative example without pruning hypotheses that are later made inconsistent or duplicated, so the raw printed `G` list contains repeated/inconsistent entries. The two boundaries above are the ones that survive as genuinely maximally-general and consistent with all examples; deduplicating and filtering `general` (standard Candidate Elimination removes any hypothesis in G more general than another, and any inconsistent with a later positive example) would be a good follow-up cleanup.

---

# 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Environment:** Jupyter Notebook

---

# ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/BhardwajYash18/ML-Practicals.git
cd ML-Practicals

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Launch Jupyter and open the desired lab notebook
jupyter notebook
```

---

# 📁 Repository Structure

```text
ML-Practicals/
├── Lab-1_Data-Preprocessing/
│   ├── YashBhardwaj_ML-1.ipynb
│   └── titanic.csv
├── Lab-2_Linear-Regression-Models/
│   ├── YashBhardwaj_Practical_2_MLR.ipynb
│   └── USA_Housing.csv
├── Lab-3_Find-S_And_CandidateElimination_Algorithm/
│   ├── A-A1-09-YashBhardwaj_ML-3.ipynb
│   └── enjoy.csv
└── README.md
```
