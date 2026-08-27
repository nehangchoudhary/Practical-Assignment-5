# Import necessary libraries
import pandas as pd
import numpy as np

# scikit-learn modules for preprocessing, models, and evaluation
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix

# ==============================================================================
# Part 1: Data Preprocessing Pipeline
# ==============================================================================
print("--- Part 1: Data Preprocessing ---")

# Load the built-in California Housing dataset
california_data = fetch_california_housing()

# Convert the dataset into a pandas DataFrame
df_california = pd.DataFrame(california_data.data, columns=california_data.feature_names)
df_california['Target'] = california_data.target

# Check for missing values in the dataset
print(f"Total missing values before imputation: {df_california.isnull().sum().sum()}")

# Separate features (X) and the target variable (y)
X_reg = df_california.drop('Target', axis=1)
y_reg = df_california['Target']

# Initialize SimpleImputer to fill any missing values using the median strategy
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_reg)

# Scale the numerical features using StandardScaler (Mean = 0, Variance = 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
print("Data successfully imputed and scaled.\n")


# ==============================================================================
# Part 2: Regression Task - Price Prediction
# ==============================================================================
print("--- Part 2: Regression Task ---")

# Split the preprocessed dataset into training (80%) and testing (20%) sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)

# Initialize the regression models
linear_model = LinearRegression()
rf_regressor = RandomForestRegressor(random_state=42)

# Train both models on the training data
linear_model.fit(X_train_reg, y_train_reg)
rf_regressor.fit(X_train_reg, y_train_reg)

# Generate price predictions on the test set
lin_preds = linear_model.predict(X_test_reg)
rf_preds_reg = rf_regressor.predict(X_test_reg)

# Function to evaluate and print regression metrics
def evaluate_regression(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Evaluation:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}\n")

# Evaluate both models
evaluate_regression(y_test_reg, lin_preds, "Linear Regression")
evaluate_regression(y_test_reg, rf_preds_reg, "Random Forest Regressor")


# ==============================================================================
# Part 3: Classification Task - Binary Classification
# ==============================================================================
print("--- Part 3: Classification Task ---")

# Load the built-in Breast Cancer Wisconsin dataset
cancer_data = load_breast_cancer()
X_clf = cancer_data.data
y_clf = cancer_data.target

# Split the data into training and testing sets (80/20 split)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Initialize the classification models
# Note: max_iter is increased for LogisticRegression to ensure it converges
logistic_model = LogisticRegression(max_iter=10000, random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)

# Train the models
logistic_model.fit(X_train_clf, y_train_clf)
rf_classifier.fit(X_train_clf, y_train_clf)

# Generate class predictions on the test set
log_preds = logistic_model.predict(X_test_clf)
rf_preds_clf = rf_classifier.predict(X_test_clf)

# Evaluate Logistic Regression
print("Logistic Regression - Confusion Matrix:")
print(confusion_matrix(y_test_clf, log_preds))
print("Logistic Regression - Classification Report:")
print(classification_report(y_test_clf, log_preds))

# Evaluate Random Forest Classifier
print("Random Forest Classifier - Confusion Matrix:")
print(confusion_matrix(y_test_clf, rf_preds_clf))
print("Random Forest Classifier - Classification Report:")
print(classification_report(y_test_clf, rf_preds_clf))


# ==============================================================================
# Part 4: Model Tuning & Cross-Validation
# ==============================================================================
print("--- Part 4: Model Tuning & Cross-Validation ---")

# Perform 5-fold cross-validation on the RandomForestClassifier
cv_scores = cross_val_score(rf_classifier, X_train_clf, y_train_clf, cv=5, scoring='accuracy')

# Print the mean accuracy across all 5 folds
print(f"5-Fold Cross-Validation Accuracies: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}\n")

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

# Initialize GridSearchCV to tune the Random Forest model
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,               # 5-fold cross-validation during tuning
    scoring='accuracy',
    n_jobs=-1           # Use all available CPU cores
)

# Fit the GridSearchCV on the training data
grid_search.fit(X_train_clf, y_train_clf)

# Print the best parameter combination found
print("GridSearchCV Hyperparameter Tuning Complete.")
print(f"Best Parameter Combination Found: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy from Grid Search: {grid_search.best_score_:.4f}")