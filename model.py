import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
#from imblearn.over_sampling import SMOTE

#load the csv file
train_dataset = pd.read_csv("/Users/tanvir/Desktop/ML_Project/Obesity_Prediction_Project/Dataset/train.csv")
test_dataset = pd.read_csv("/Users/tanvir/Desktop/ML_Project/Obesity_Prediction_Project/Dataset/test.csv")

print(train_dataset.head())
print(test_dataset.head())

#Removing ID
ids = test_dataset['id']
test_dataset = test_dataset.drop('id', axis=1)
train_dataset = train_dataset.drop('id', axis=1)
print(test_dataset.head())
print(train_dataset.head())

#Data Preprocessing
scaler = MinMaxScaler()
num_cols = train_dataset.select_dtypes(include='number').columns
train_dataset[num_cols] = scaler.fit_transform(train_dataset[num_cols])
test_dataset[num_cols] = scaler.transform(test_dataset[num_cols])

print(train_dataset.head())

#Encoding
#encoding
encoder = LabelEncoder()
categorical_col = train_dataset.drop('NObeyesdad', axis=1).select_dtypes(include='object').columns
train_dataset[categorical_col] = train_dataset[categorical_col].apply(encoder.fit_transform)
test_dataset[categorical_col] = test_dataset[categorical_col].apply(encoder.fit_transform)

print(test_dataset.head())

#Train Test Split
X = train_dataset.drop('NObeyesdad', axis=1)
y = train_dataset['NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize SMOTE with a specified random state
#X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# Optionally, check the class distribution before and after oversampling
#print("Class distribution before oversampling:", y_train.value_counts())
#print("Class distribution after oversampling:", y_train_resampled.value_counts())

#MOdel Implementation
best_params = {
    'bootstrap': True,
    'criterion': 'gini',
    'max_depth': 10,
    'max_features': 0.48231592334473916,
    'min_samples_leaf': 1,
    'min_samples_split': 15,
    'n_estimators': 100
}
best_rf = RandomForestClassifier(**best_params, random_state=66)
best_rf.fit(X_train, y_train)

#Make a pickle file of the model
pickle.dump(best_rf, open("model.pkl", "wb"))

y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)
preds_proba = best_rf.predict_proba(test_dataset)
Prediction = np.argmax(preds_proba, axis=1)
submission_df = pd.DataFrame({"id": ids, "NObeyesdad":Prediction})



