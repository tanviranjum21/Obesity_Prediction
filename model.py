import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

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
        ids = test_dataset['id']
        return train_dataset, test_dataset, ids

    def preprocess_data(self, train_dataset, test_dataset):
        """
        Preprocess the train and test datasets.

        Args:
        train_dataset (DataFrame): The training dataset.
        test_dataset (DataFrame): The test dataset.

        Returns:
        tuple: A tuple containing the preprocessed training dataset and the preprocessed test dataset.
        """
        # Combine train and test datasets for preprocessing
        combined_dataset = pd.concat([train_dataset, test_dataset], axis=0)

        # Remove ID column
        combined_dataset = combined_dataset.drop('id', axis=1)

        # Feature Engineering
        combined_dataset['BMI'] = combined_dataset['Weight'] / (combined_dataset['Height'] ** 2)

        # Encode categorical features
        categorical_cols = combined_dataset.select_dtypes(include='object').columns
        for col in categorical_cols:
            self.encoder.fit(combined_dataset[col])
            combined_dataset[col] = self.encoder.transform(combined_dataset[col])

        # Split back into train and test datasets
        train_len = len(train_dataset)
        train_dataset = combined_dataset[:train_len]
        test_dataset = combined_dataset[train_len:]

        return train_dataset, test_dataset

    def train_model(self, X_train, y_train):
        """
        Train a Gradient Boosting model.

        Args:
        X_train (DataFrame): Features of the training dataset.
        y_train (Series): Labels of the training dataset.

        Returns:
        GradientBoostingClassifier: Trained Gradient Boosting model.
        """
        model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        return model

    def save_model(self, model, file_path):
        """
        Save the trained model to a file.

        Args:
        model (GradientBoostingClassifier): Trained Gradient Boosting model.
        file_path (str): File path to save the model.
        """
        with open(file_path, "wb") as f:
            pickle.dump(model, f)

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the trained model.

        Args:
        model (GradientBoostingClassifier): Trained Gradient Boosting model.
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
        model (GradientBoostingClassifier): Trained Gradient Boosting model.
        test_dataset (DataFrame): Test dataset.
        ids (array-like): Array of IDs for the test dataset.

        Returns:
        DataFrame: DataFrame containing predictions.
        """
        # Align the columns of the test dataset with the training dataset
        test_dataset = test_dataset.reindex(columns=X.columns, fill_value=0)

        predictions = model.predict(test_dataset)
        submission_df = pd.DataFrame({"id": ids, "NObeyesdad": predictions})
        return submission_df

if __name__ == "__main__":
    predictor = ObesityPredictor()
    train_dataset, test_dataset, ids = predictor.load_data(TRAIN_FILE_PATH, TEST_FILE_PATH)
    train_dataset, test_dataset = predictor.preprocess_data(train_dataset, test_dataset)
    X = train_dataset.drop('NObeyesdad', axis=1)
    y = train_dataset['NObeyesdad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    predictor.model = predictor.train_model(X_train, y_train)
    predictor.save_model(predictor.model, MODEL_FILE_PATH)

    accuracy, classification_rep = predictor.evaluate_model(predictor.model, X_test, y_test)
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_rep)

    submission_df = predictor.predict(predictor.model, test_dataset, ids)
    print(submission_df.head())

    decoding_mapping = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II",
    }
    submission_df["NObeyesdad"] = submission_df["NObeyesdad"].map(decoding_mapping)

    # Saving Submission File
    submission_df.to_csv('submission.csv', index=False)
