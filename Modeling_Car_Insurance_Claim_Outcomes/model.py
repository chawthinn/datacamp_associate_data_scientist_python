import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import cross_val_score

# Load dataset
df = pd.read_csv("Modeling_Car_Insurance_Claim_Outcomes/transformed_car_insurance.csv")

# Define feature columns and target
X = df.drop(columns=['id','outcome', 'postal_code'])  # Features excluding id
y = df['outcome']  # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training size: {X_train.shape}, Test size: {X_test.shape}")

# Initialize LazyClassifier
clf = LazyClassifier(ignore_warnings=True, random_state=42)

# Run models and get performance metrics
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Display the results
print(models)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

# Select the best model: QuadraticDiscriminantAnalysis
best_model = QuadraticDiscriminantAnalysis()

# Perform cross-validation
scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# Define models
models = {
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
    'LGBMClassifier': LGBMClassifier(random_state=42),
    'LinearSVC': LinearSVC(random_state=42, max_iter=10000)  # Increase max_iter to avoid convergence issues
}

# Perform cross-validation for each model
results = []
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results.append({
        'Model': name,
        'Mean Accuracy': scores.mean(),
        'Std Dev': scores.std()
    })

# Create a DataFrame for comparison
results_df = pd.DataFrame(results)

# Display results
print(results_df)

from lightgbm import LGBMClassifier
import joblib

# Train the model
lgbm_model = LGBMClassifier(random_state=42)
lgbm_model.fit(X_train, y_train)

# Extract feature importance
importances = lgbm_model.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)

# Save the model
joblib.dump(lgbm_model, 'Modeling_Car_Insurance_Claim_Outcomes/lgbm_model.pkl')
