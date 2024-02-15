import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Constants
TRAIN_FILE_PATH = "/Users/tanvir/Desktop/ML_Project/Obesity_Prediction_Project/Dataset/train.csv"
TEST_FILE_PATH = "/Users/tanvir/Desktop/ML_Project/Obesity_Prediction_Project/Dataset/test.csv"
MODEL_FILE_PATH = "model.pkl"


class ObesityPredictor:
    def __init__(self):
        """
        Initialize ObesityPredictor object.
        """
        self.model = None
        self.encoder = LabelEncoder()
        self.scaler = MinMaxScaler()

    def load_data(self, train_file_path, test_file_path):
        """
        Load train and test datasets.

        Args:
        train_file_path (str): File path to the training dataset.
        test_file_path (str): File path to the test dataset.

        Returns:
        tuple: A tuple containing the training dataset and the test dataset.
        """
        train_dataset = pd.read_csv(train_file_path)
        test_dataset = pd.read_csv(test_file_path)
        return train_dataset, test_dataset

    def preprocess_data(self, train_dataset, test_dataset):
        """
        Preprocess the train and test datasets.

        Args:
        train_dataset (DataFrame): The training dataset.
        test_dataset (DataFrame): The test dataset.

        Returns:
        tuple: A tuple containing the preprocessed training dataset and the preprocessed test dataset.
        """
        # Remove ID column
        train_dataset = train_dataset.drop('id', axis=1)
        test_dataset = test_dataset.drop('id', axis=1)

        # Scale numeric features
        num_cols = train_dataset.select_dtypes(include='number').columns
        train_dataset[num_cols] = self.scaler.fit_transform(train_dataset[num_cols])
        test_dataset[num_cols] = self.scaler.transform(test_dataset[num_cols])

        # Encode categorical features
        categorical_cols = train_dataset.drop('NObeyesdad', axis=1).select_dtypes(include='object').columns
        train_dataset[categorical_cols] = train_dataset[categorical_cols].apply(self.encoder.fit_transform)
        test_dataset[categorical_cols] = test_dataset[categorical_cols].apply(self.encoder.fit_transform)

        return train_dataset, test_dataset

    def train_model(self, X_train, y_train, params):
        """
        Train a Random Forest model.

        Args:
        X_train (DataFrame): Features of the training dataset.
        y_train (Series): Labels of the training dataset.
        params (dict): Parameters for the Random Forest model.

        Returns:
        RandomForestClassifier: Trained Random Forest model.
        """
        model = RandomForestClassifier(**params, random_state=66)
        model.fit(X_train, y_train)
        return model

    def save_model(self, model, file_path):
        """
        Save the trained model to a file.

        Args:
        model (RandomForestClassifier): Trained Random Forest model.
        file_path (str): File path to save the model.
        """
        with open(file_path, "wb") as f:
            pickle.dump(model, f)

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the trained model.

        Args:
        model (RandomForestClassifier): Trained Random Forest model.
        X_test (DataFrame): Features of the test dataset.
        y_test (Series): Labels of the test dataset.

        Returns:
        tuple: A tuple containing the accuracy score and classification report.
        """
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        return accuracy, classification_rep

    def predict(self, model, test_dataset, ids):
        """
        Make predictions using the trained model.

        Args:
        model (RandomForestClassifier): Trained Random Forest model.
        test_dataset (DataFrame): Test dataset.
        ids (array-like): Array of IDs for the test dataset.

        Returns:
        DataFrame: DataFrame containing predictions.
        """
        preds_proba = model.predict_proba(test_dataset)
        predictions = np.argmax(preds_proba, axis=1)
        submission_df = pd.DataFrame({"id": ids, "NObeyesdad": predictions})
        return submission_df


if __name__ == "__main__":
    predictor = ObesityPredictor()
    train_dataset, test_dataset = predictor.load_data(TRAIN_FILE_PATH, TEST_FILE_PATH)
    train_dataset, test_dataset = predictor.preprocess_data(train_dataset, test_dataset)
    X = train_dataset.drop('NObeyesdad', axis=1)
    y = train_dataset['NObeyesdad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    best_params = {
        'bootstrap': True,
        'criterion': 'gini',
        'max_depth': 10,
        'max_features': 0.48231592334473916,
        'min_samples_leaf': 1,
        'min_samples_split': 15,
        'n_estimators': 100
    }

    predictor.model = predictor.train_model(X_train, y_train, best_params)
    predictor.save_model(predictor.model, MODEL_FILE_PATH)

    accuracy, classification_rep = predictor.evaluate_model(predictor.model, X_test, y_test)
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_rep)

    submission_df = predictor.predict(predictor.model, test_dataset, test_dataset.index)
    print(submission_df.head())
