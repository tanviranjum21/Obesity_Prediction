from config import *
from model import ObesityPredictor, TRAIN_FILE_PATH, TEST_FILE_PATH, MODEL_FILE_PATH
from flask import Flask, request, jsonify

app = Flask(__name__)
predictor = ObesityPredictor()

# Load and preprocess data
train_dataset, test_dataset, ids = predictor.load_data(TRAIN_FILE_PATH, TEST_FILE_PATH)
train_dataset, test_dataset = predictor.preprocess_data(train_dataset, test_dataset)
X = train_dataset.drop('NObeyesdad', axis=1)
y = train_dataset['NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train model
grid_search = predictor.train_model(X_train, y_train)
predictor.save_model(grid_search, MODEL_FILE_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    test_dataset = pd.DataFrame(data)
    submission_df = predictor.predict(grid_search, test_dataset, ids)
    return jsonify(submission_df.to_dict())

if __name__ == '__main__':
    app.run(debug=True)
