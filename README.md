This project applies machine learning to identify trustworthy NGOs, enabling donors to make informed and transparent giving decisions.
By analyzing key indicators such as organizational age, fund utilization ratio, public rating, social media followers, and location, 
the system predicts the trustworthiness of an NGO using a data-driven and unbiased approach.
📌 Features
Synthetic NGO dataset with relevant trust indicators.
Implements Random Forest, Logistic Regression, and K-Nearest Neighbors classifiers.
Performance evaluation using accuracy, precision, recall, and confusion matrix.
Comparative analysis to determine the most effective model (Random Forest performed best).
Encourages transparency, accountability, and informed decision-making in the NGO sector.

🛠 Methodology
Data Collection & Preparation – A synthetic dataset was prepared with NGO attributes.
Feature Selection – Age, fund utilization ratio, public rating, follower count, and location.
Model Training – Trained three ML models:
Random Forest (best performance, lowest false negatives)
Logistic Regression (baseline model)
KNN (distance-based classification)
Model Evaluation – Metrics: Accuracy, Precision, Recall, and Confusion Matrix.

📊 Results
Random Forest achieved the highest accuracy and best balance of precision/recall.
Public rating and fund utilization ratio were the most influential features.
The approach can be expanded with real datasets and additional features for better generalization
