
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import gdown

google_drive_link = "https://drive.google.com/file/d/1a-0o_X3QscN6CnEKmAFpQ2QLxLEMvuPG/view?usp=sharing"

file_id = google_drive_link.split("/")[-2]
download_url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(download_url, "dataset.csv", quiet=False)
data = pd.read_csv("dataset.csv")

data = pd.get_dummies(data, columns=['Location', 'Certified(Year/N)'], drop_first=True)
data = data.drop(columns=['NGOName', 'RegistrationiD', 'ngoReviews'])

X = data.drop('TrustLabel', axis=1)
y = data['TrustLabel']

X = data.drop('TrustLabel', axis=1)
y = data['TrustLabel']

minority_class_count = y.value_counts().min()
k_neighbors = min(5, minority_class_count - 1)  # Dynamic k_neighbors
if k_neighbors >= 1:
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)
else:
    print("Not enough samples in the minority class to apply SMOTE.")
    X_resampled, y_resampled = X, y

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.3, random_state=42)

def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Trustable", "Trustworthy"], yticklabels=["Not Trustable", "Trustworthy"])
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def plot_feature_importance(model, X, title):
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
        features = X.columns
        sorted_indices = np.argsort(feature_importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(features)), feature_importances[sorted_indices], align='center')
        plt.xticks(range(len(features)), np.array(features)[sorted_indices], rotation=90)
        plt.title(title)
        plt.show()


#  here we doing the Random Forest with Hyperparameter Tuning
rf_model = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
grid_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_train, y_train)
rf_best = grid_rf.best_estimator_
rf_pred = rf_best.predict(X_test)
print("\nRandom Forest Results:")
print("Best Parameters:", grid_rf.best_params_)
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))
print("Accuracy:", accuracy_score(y_test, rf_pred))
plot_confusion_matrix(y_test, rf_pred, "Random Forest")
plot_feature_importance(rf_best, pd.DataFrame(X_resampled, columns=X.columns), "Random Forest Feature Importance")

# here we are doing Logistic Regression with Class Weights
lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=200)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("\nLogistic Regression Results:")
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print("Classification Report:\n", classification_report(y_test, lr_pred))
print("Accuracy:", accuracy_score(y_test, lr_pred))
plot_confusion_matrix(y_test, lr_pred, "Logistic Regression")

# here we are using K-Nearest Neighbors with Hyperparameter Tuning
knn_model = KNeighborsClassifier()
param_grid_knn = {'n_neighbors': [3, 5, 7, 9]}
grid_knn = GridSearchCV(knn_model, param_grid_knn, cv=5, scoring='accuracy')
grid_knn.fit(X_train, y_train)
knn_best = grid_knn.best_estimator_
knn_pred = knn_best.predict(X_test)
print("\nK-Nearest Neighbors Results:")
print("Best Parameters:", grid_knn.best_params_)
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("Classification Report:\n", classification_report(y_test, knn_pred))
print("Accuracy:", accuracy_score(y_test, knn_pred))
plot_confusion_matrix(y_test, knn_pred, "K-Nearest Neighbors")

#  OUR NEW NGO DATA PREDICTION

# Define new NGO data (Example data, can be changed as needed)
new_data = {
    "Age (Years)": [30],
    "FundUtilizeratio": [0.75],
    "NumofCompletedProjects": [25],
    "PublicRating (1-5)": [3.27],
    "SocialMediaFollower": [7000],
    "Location": ["Dhaka_North"],
    "Certified(Year/N)": ["Y"],
}

new_data_df = pd.DataFrame(new_data)

def preprocess_new_data(new_data, dummies_columns):
    new_data = pd.get_dummies(new_data, columns=dummies_columns, drop_first=True)
    missing_cols = set(X.columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0
    # HERE WE ARE Reorder columns to match the training data
    new_data = new_data[X.columns]
    return new_data

new_data_df = preprocess_new_data(new_data_df, dummies_columns=['Location', 'Certified(Year/N)'])

new_data_scaled = scaler.transform(new_data_df)

rf_prediction = rf_best.predict(new_data_scaled)
lr_prediction = lr_model.predict(new_data_scaled)
knn_prediction = knn_best.predict(new_data_scaled)

# Displaying predictions for new NGO
print("\nTrustworthiness Predictions for New NGO:")
print(f"Random Forest: {'Trustworthy' if rf_prediction[0] else 'Not Trustable'}")
print(f"Logistic Regression: {'Trustworthy' if lr_prediction[0] else 'Not Trustable'}")
print(f"K-Nearest Neighbors: {'Trustworthy' if knn_prediction[0] else 'Not Trustable'}")
