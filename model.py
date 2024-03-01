from config import *
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
        Train a XGBoost model with hyperparameter tuning using GridSearchCV.

        Args:
        X_train (DataFrame): Features of the training dataset.
        y_train (Series): Labels of the training dataset.

        Returns:
        GridSearchCV: Trained GridSearchCV object.
        """
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5]
        }

        model = xgb.XGBClassifier(objective='multi:softmax', num_class=7, random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        return grid_search

    def save_model(self, model, file_path):
        """
        Save the trained model to a file.

        Args:
        model (GridSearchCV): Trained GridSearchCV object.
        file_path (str): File path to save the model.
        """
        with open(file_path, "wb") as f:
            pickle.dump(model, f)

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the trained model.

        Args:
        model (GridSearchCV): Trained GridSearchCV object.
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
        model (GridSearchCV): Trained GridSearchCV object.
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
