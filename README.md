# Task5_AI_ML_Internship


## Task 5: Decision Trees and Random Forests

## Objective
To understand and implement tree-based machine learning models for classification using Decision Tree and Random Forest, analyze overfitting, interpret feature importance, and evaluate performance using cross-validation.

## Dataset
Heart Disease Dataset

The dataset contains medical attributes used to predict whether a patient has heart disease (target variable).

- Target: 0 = No Disease
- Target: 1 = Disease
- Features include age, cholesterol, blood pressure, chest pain type, etc.

## Tools & Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Steps Performed

### 1️⃣ Data Preprocessing

- Loaded dataset using pandas
- Checked for missing values
- Split data into features (X) and target (y)
- Performed train-test split (80:20)

### 2️⃣ Decision Tree Model

- Trained DecisionTreeClassifier
- Evaluated accuracy
- Visualized the tree structure
- Analyzed overfitting by comparing train and test accuracy

### 3️⃣ Overfitting Control

- Controlled tree depth using max_depth
- Used min_samples_split
- Observed improvement in generalization

### 4️⃣ Random Forest Model

- Trained RandomForestClassifier
- Compared accuracy with Decision Tree
- Observed better performance due to ensemble learning

### 5️⃣ Feature Importance

- Extracted feature importance scores
- Plotted bar graph to interpret influential features
- Identified most important medical factors contributing to prediction

### 6️⃣ Cross-Validation

- Applied 5-fold cross-validation
- Calculated average accuracy for reliable model evaluation

## Results

- Random Forest performed better than a single Decision Tree
- Controlled tree depth reduced overfitting
- Feature importance helped identify key predictors

## Conclusion

Tree-based models are powerful and interpretable machine learning techniques. While Decision Trees are easy to visualize and understand, Random Forest improves accuracy and reduces overfitting by combining multiple trees.

