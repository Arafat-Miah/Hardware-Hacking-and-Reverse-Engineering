import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Adjust the number of estimators or other parameters if necessary
    models = [RandomForestClassifier(n_estimators=100) for _ in range(response_data.shape[1])]  # Changed to RandomForestClassifier
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "E:\\Study Oulu\\2. Second period\\Hardware Hacing\\Challenge.txt"
y_text_file = "E:\\Study Oulu\\2. Second period\\Hardware Hacing\\Response.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (85/15 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.15, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
correct_predictions = 0
total_tests = len(X_test)

for idx in range(total_tests):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    if np.array_equal(predicted_response, actual_response):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_tests) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests}")
print(f"Overall accuracy: {accuracy:.2f}%")





