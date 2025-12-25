#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.naive_bayes import BernoulliNB

def prepare_data(challenges, responses):
    # Convert challenge and response strings to numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in challenges])
    response_data = np.array([[int(bit) for bit in response] for response in responses])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Initialize a model for each bit in the response
    models = [BernoulliNB() for _ in range(response_data.shape[1])]
    # Train each model
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, new_challenge):
    # Prepare the new challenge for prediction
    new_challenge_array = np.array([[int(bit) for bit in new_challenge]])
    # Predict each bit using the corresponding model
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return ''.join(map(str, predicted_response))

# Example data
challenges = [
    "000001101100100101101000",
    "000001100111000100001100",
    "010111111010010001010111",
    "111011100110101101010110",
    "101100110101110001111000"
]
responses = [
    "01001000001111111000000001111100010000100011101000110101110001110000010110101110100110100100111100111111011010001100000100011011",
    "01010010001101101001101100001001010000100011101000110101110001110011000110101011101010000101010100111101110001000011011101110110",
    "01000000011011111111110011111001001111100101100000111101110111110011100010000000101011011010000000111100011010111001101000001000",
    "00111110101111011101111111011000001111010010001000001010010000100100010110011011011100010110101001000101111001001001001110001000",
    "00111011100110101100101000000000001110111110111100101101001100010011110110001011111000110010011100111101001000100000101001000010"
]

# Prepare data and train the model
challenge_data, response_data = prepare_data(challenges, responses)
models = train_models(challenge_data, response_data)

# Predict a new response
new_challenge = "000001111000011010011011"
predicted_response = predict_response(models, new_challenge)
print("Predicted Response:", predicted_response)


# In[3]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB

def prepare_data(x_df, y_df):
    # Convert DataFrame columns containing binary string data into numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Initialize a model for each bit in the response
    models = [BernoulliNB() for _ in range(response_data.shape[1])]
    # Train each model
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, new_challenge):
    # Prepare the new challenge for prediction
    new_challenge_array = np.array([[int(bit) for bit in new_challenge]])
    # Predict each bit using the corresponding model
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return ''.join(map(str, predicted_response))

# Load challenge data from file
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_100.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])

# Load response data from file
y_text_file = "D:\\data preperation\\New folder\\selected_responses_100.txt"
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data and train the model
challenge_data, response_data = prepare_data(x_df, y_df)
models = train_models(challenge_data, response_data)

# Predict a new response
new_challenge = "100010011000001111100001"
predicted_response = predict_response(models, new_challenge)
print("Predicted Response:", predicted_response)


# In[5]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

def prepare_data(x_df, y_df):
    # Convert DataFrame columns containing binary string data into numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Initialize a model for each bit in the response
    models = [BernoulliNB() for _ in range(response_data.shape[1])]
    # Train each model
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    # Predict each bit using the corresponding model
    predicted_response = np.array([model.predict(challenge) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_100.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_100.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42)

# Train models on training data
models = train_models(X_train, y_train)

# Choose a random index from the test set to simulate user input
test_index = np.random.randint(0, len(X_test))  # Get a random index from the test set

predicted_response = predict_response(models, X_test[test_index].reshape(1, -1))
actual_response = y_test[test_index]

# Calculate the number of bits that match and do not match
matches = np.sum(predicted_response == actual_response)
mismatches = len(predicted_response) - matches

print(f"Test Index: {test_index}")
print(f"Challenge Used for Prediction: {x_df.iloc[X_test.index[test_index]]['Challenge']}")
print(f"Predicted Response: {''.join(map(str, predicted_response))}")
print(f"Actual Response:    {''.join(map(str, actual_response))}")
print(f"Number of matching bits: {matches}")
print(f"Number of non-matching bits: {mismatches}")


# In[8]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

def prepare_data(x_df, y_df):
    # Convert DataFrame columns containing binary string data into numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Initialize a model for each bit in the response
    models = [BernoulliNB() for _ in range(response_data.shape[1])]
    # Train each model
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, new_challenge):
    # Prepare the new challenge for prediction
    new_challenge_array = np.array([[int(bit) for bit in new_challenge]])
    # Predict each bit using the corresponding model
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return ''.join(map(str, predicted_response))

def find_actual_response(x_df, y_df, challenge):
    # Look for the challenge in the DataFrame and get the corresponding response if found
    match = x_df[x_df['Challenge'] == challenge].index.tolist()
    if match:
        response = y_df.iloc[match[0]]['Y_Data']
        return response
    else:
        return None

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_100.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_100.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42)

# Train models on training data
models = train_models(X_train, y_train)

# User-provided challenge for prediction
user_challenge = "110111011011101101011101"
predicted_response = predict_response(models, user_challenge)
actual_response = find_actual_response(x_df, y_df, user_challenge)

print(f"User Given Challenge: {user_challenge}")
print(f"Predicted Response: {predicted_response}")
if actual_response:
    print(f"Actual Response: {actual_response}")
else:
    print("No actual response found for the given challenge in the dataset.")


# In[10]:


#serial test train nei na
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

def prepare_data(x_df, y_df):
    # Convert DataFrame columns containing binary string data into numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Initialize a model for each bit in the response
    models = [BernoulliNB() for _ in range(response_data.shape[1])]
    # Train each model
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, new_challenge):
    # Prepare the new challenge for prediction
    new_challenge_array = np.array([[int(bit) for bit in new_challenge]])
    # Predict each bit using the corresponding model
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return ''.join(map(str, predicted_response))

def find_actual_response(x_df, y_df, challenge):
    # Look for the challenge in the DataFrame and get the corresponding response if found
    match = x_df[x_df['Challenge'] == challenge].index.tolist()
    if match:
        response = y_df.iloc[match[0]]['Y_Data']
        return response
    else:
        return None

def count_bit_matches(predicted, actual):
    # Count the number of matching and non-matching bits
    matches = sum(1 for x, y in zip(predicted, actual) if x == y)
    mismatches = len(predicted) - matches
    return matches, mismatches

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_100.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_100.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42)

# Train models on training data
models = train_models(X_train, y_train)

# User-provided challenge for prediction
user_challenge = "000001111000011010011011"
predicted_response = predict_response(models, user_challenge)
actual_response = find_actual_response(x_df, y_df, user_challenge)

print(f"User Given Challenge: {user_challenge}")
print(f"Predicted Response: {predicted_response}")
if actual_response:
    print(f"Actual Response: {actual_response}")
    matches, mismatches = count_bit_matches(predicted_response, actual_response)
    print(f"Number of matching bits: {matches}")
    print(f"Number of non-matching bits: {mismatches}")
else:
    print("No actual response found for the given challenge in the dataset.")


# In[12]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

def prepare_data(x_df, y_df):
    # Convert DataFrame columns containing binary string data into numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Initialize a model for each bit in the response
    models = [BernoulliNB() for _ in range(response_data.shape[1])]
    # Train each model
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, new_challenge):
    # Prepare the new challenge for prediction
    new_challenge_array = np.array([[int(bit) for bit in new_challenge]])
    # Predict each bit using the corresponding model
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return ''.join(map(str, predicted_response))

def find_actual_response(x_df, y_df, challenge):
    # Look for the challenge in the DataFrame and get the corresponding response if found
    match = x_df[x_df['Challenge'] == challenge].index.tolist()
    if match:
        response = y_df.iloc[match[0]]['Y_Data']
        return response
    else:
        return None

def count_bit_matches(predicted, actual):
    # Count the number of matching and non-matching bits
    matches = sum(1 for x, y in zip(predicted, actual) if x == y)
    mismatches = len(predicted) - matches
    return matches, mismatches

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_1000.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_1000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split), with no shuffling to preserve order
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=False)

# Train models on training data
models = train_models(X_train, y_train)

# User-provided challenge for prediction
user_challenge = "100001100000000110101011"
predicted_response = predict_response(models, user_challenge)
actual_response = find_actual_response(x_df, y_df, user_challenge)

print(f"User Given Challenge: {user_challenge}")
print(f"Predicted Response: {predicted_response}")
if actual_response:
    print(f"Actual Response: {actual_response}")
    matches, mismatches = count_bit_matches(predicted_response, actual_response)
    print(f"Number of matching bits: {matches}")
    print(f"Number of non-matching bits: {mismatches}")
else:
    print("No actual response found for the given challenge in the dataset.")


# In[16]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

def prepare_data(x_df, y_df):
    # Convert DataFrame columns containing binary string data into numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Initialize a model for each bit in the response
    models = [BernoulliNB() for _ in range(response_data.shape[1])]
    # Train each model
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, new_challenge):
    # Prepare the new challenge for prediction
    new_challenge_array = np.array([[int(bit) for bit in new_challenge]])
    # Predict each bit using the corresponding model
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return ''.join(map(str, predicted_response))

def find_actual_response(x_df, y_df, challenge):
    # Look for the challenge in the DataFrame and get the corresponding response if found
    match = x_df[x_df['Challenge'] == challenge].index.tolist()
    if match:
        response = y_df.iloc[match[0]]['Y_Data']
        return response
    else:
        return None

def count_bit_matches(predicted, actual):
    # Count the number of matching and non-matching bits
    matches = sum(1 for x, y in zip(predicted, actual) if x == y)
    mismatches = len(predicted) - matches
    return matches, mismatches

def calculate_accuracy(matches, total_bits):
    # Calculate accuracy as the percentage of matching bits
    return (matches / total_bits) * 100

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_1000.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_1000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split), with no shuffling to preserve order
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=False)

# Train models on training data
models = train_models(X_train, y_train)

# User-provided challenge for prediction
user_challenge = "100001100000000110101011"
predicted_response = predict_response(models, user_challenge)
actual_response = find_actual_response(x_df, y_df, user_challenge)

print(f"User Given Challenge: {user_challenge}")
print(f"Predicted Response: {predicted_response}")
if actual_response:
    print(f"Actual Response: {actual_response}")
    matches, mismatches = count_bit_matches(predicted_response, actual_response)
    accuracy = calculate_accuracy(matches, len(actual_response))
    print(f"Number of matching bits: {matches}")
    print(f"Number of non-matching bits: {mismatches}")
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("No actual response found for the given challenge in the dataset.")


# In[17]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

def prepare_data(x_df, y_df):
    # Convert DataFrame columns containing binary string data into numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Initialize a model for each bit in the response
    models = [BernoulliNB() for _ in range(response_data.shape[1])]
    # Train each model
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    # Predict each bit using the corresponding model
    new_challenge_array = np.array([[int(bit) for bit in challenge]])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return ''.join(map(str, predicted_response))

def calculate_accuracy(matches, total_bits):
    # Calculate accuracy as the percentage of matching bits
    return (matches / total_bits) * 100

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_100.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_100.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=False)

# Train models on training data
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
total_matches = 0
total_mismatches = 0
total_bits = 0

for idx in range(len(X_test)):
    test_challenge = X_test[idx]
    actual_response = ''.join(map(str, y_test[idx]))
    predicted_response = predict_response(models, test_challenge)
    matches, mismatches = sum(p == a for p, a in zip(predicted_response, actual_response)), sum(p != a for p, a in zip(predicted_response, actual_response))

    total_matches += matches
    total_mismatches += mismatches
    total_bits += len(actual_response)

    print(f"Test Challenge {idx}:")
    print(f"Predicted: {predicted_response}")
    print(f"Actual:    {actual_response}")
    print(f"Matches: {matches}, Mismatches: {mismatches}")

# Calculate total accuracy
total_accuracy = calculate_accuracy(total_matches, total_bits)
print(f"\nTotal matches: {total_matches}, Total mismatches: {total_mismatches}, Total bits: {total_bits}")
print(f"Overall accuracy: {total_accuracy:.2f}%")


# In[20]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

def prepare_data(x_df, y_df):
    # Convert DataFrame columns containing binary string data into numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Initialize a model for each bit in the response
    models = [BernoulliNB() for _ in range(response_data.shape[1])]
    # Train each model
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    # Predict each bit using the corresponding model
    new_challenge_array = np.array([[int(bit) for bit in challenge]])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return ''.join(map(str, predicted_response))

def calculate_accuracy(matches, total_bits):
    # Calculate accuracy as the percentage of matching bits
    return (matches / total_bits) * 100

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_10000.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_10000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=False)

# Train models on training data
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
total_matches = 0
total_mismatches = 0
total_bits = 0

for idx in range(len(X_test)):
    test_challenge = X_test[idx]
    actual_response = ''.join(map(str, y_test[idx]))
    predicted_response = predict_response(models, test_challenge)
    matches = sum(p == a for p, a in zip(predicted_response, actual_response))
    mismatches = sum(p != a for p, a in zip(predicted_response, actual_response))

    total_matches += matches
    total_mismatches += mismatches
    total_bits += len(actual_response)

# Calculate total accuracy
total_accuracy = calculate_accuracy(total_matches, total_bits)
print(f"Total matches: {total_matches}, Total mismatches: {total_mismatches}, Total bits: {total_bits}")
print(f"Overall accuracy: {total_accuracy:.2f}%")


# In[21]:


#aii khane chunk akare prediction dise.
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    # Convert DataFrame columns containing binary string data into numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    # Encode each unique binary response as a categorical label
    response_encoder = LabelEncoder()
    response_labels = response_encoder.fit_transform([''.join(map(str, response)) for response in y_df['Y_Data']])
    return challenge_data, response_labels, response_encoder

def train_model(challenge_data, response_labels):
    # Initialize a multinomial Naive Bayes model
    model = MultinomialNB()
    # Train the model on the entire challenge-response set
    model.fit(challenge_data, response_labels)
    return model

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_1000.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_1000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_labels, response_encoder = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_labels, test_size=0.2, random_state=42, shuffle=True)

# Train model on training data
model = train_model(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Decode predictions to binary strings (optional, for verification)
decoded_predictions = response_encoder.inverse_transform(y_pred)
decoded_actuals = response_encoder.inverse_transform(y_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[23]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

def prepare_data(x_df, y_df):
    # Convert DataFrame columns containing binary string data into numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Initialize a model for each bit in the response
    models = [BernoulliNB() for _ in range(response_data.shape[1])]
    # Train each model
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    # Predict each bit using the corresponding model
    new_challenge_array = np.array([[int(bit) for bit in challenge]])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return ''.join(map(str, predicted_response))

def calculate_accuracy(matches, total_bits):
    # Calculate accuracy as the percentage of matching bits
    return (matches / total_bits) * 100

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\Final\\selected_challenges1_10000.txt"
y_text_file = "D:\\data preperation\\Final\\selected_responses1_10000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=False)

# Train models on training data
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
total_matches = 0
total_mismatches = 0
total_bits = 0

for idx in range(len(X_test)):
    test_challenge = X_test[idx]
    actual_response = ''.join(map(str, y_test[idx]))
    predicted_response = predict_response(models, test_challenge)
    matches = sum(p == a for p, a in zip(predicted_response, actual_response))
    mismatches = sum(p != a for p, a in zip(predicted_response, actual_response))

    total_matches += matches
    total_mismatches += mismatches
    total_bits += len(actual_response)

# Calculate total accuracy
total_accuracy = calculate_accuracy(total_matches, total_bits)
print(f"Total matches: {total_matches}, Total mismatches: {total_mismatches}, Total bits: {total_bits}")
print(f"Overall accuracy: {total_accuracy:.2f}%")


# In[24]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    # Convert DataFrame columns containing binary string data into numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    # Encode each unique binary response as a categorical label
    response_encoder = LabelEncoder()
    response_labels = response_encoder.fit_transform([''.join(map(str, response)) for response in y_df['Y_Data']])
    return challenge_data, response_labels, response_encoder

def train_model(challenge_data, response_labels, alpha=1.0):
    # Initialize a multinomial Naive Bayes model with a custom alpha for smoothing
    model = MultinomialNB(alpha=alpha)
    # Train the model on the entire challenge-response set
    model.fit(challenge_data, response_labels)
    return model

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_1000.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_1000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_labels, response_encoder = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_labels, test_size=0.2, random_state=42, shuffle=True)

# Train model on training data with a specific alpha value
model = train_model(X_train, y_train, alpha=0.1)

# Predict on the test set
y_pred = model.predict(X_test)

# Decode predictions to binary strings (optional, for verification)
decoded_predictions = response_encoder.inverse_transform(y_pred)
decoded_actuals = response_encoder.inverse_transform(y_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Optional: Print some predictions for verification
print("\nSample Predictions:")
for i in range(5):  # Adjust the range for more or fewer samples
    print(f"Predicted: {decoded_predictions[i]}, Actual: {decoded_actuals[i]}")


# In[26]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    # Convert DataFrame columns containing binary string data into numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Initialize a model for each bit in the response
    models = [BernoulliNB() for _ in range(response_data.shape[1])]
    # Train each model on the corresponding bit of the response
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    # Predict each bit using the corresponding model
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_10000.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_10000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models on training data
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
total_matches = 0
total_bits = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    matches = np.sum(predicted_response == actual_response)
    total_matches += matches
    total_bits += len(actual_response)

# Calculate total accuracy
accuracy = (total_matches / total_bits) * 100
print(f"Total matches: {total_matches}, Total bits: {total_bits}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[29]:


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def prepare_data(x_df, y_df):
    # Convert DataFrame columns containing binary string data into numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Initialize a KNN model for each bit in the response
    models = [KNeighborsClassifier(n_neighbors=5) for _ in range(response_data.shape[1])]  # Using default 5 neighbors
    # Train each model
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    # Predict each bit using the corresponding model
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return ''.join(map(str, predicted_response))

def calculate_accuracy(matches, total_bits):
    # Calculate accuracy as the percentage of matching bits
    return (matches / total_bits) * 100

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_10000.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_10000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models on training data
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
total_matches = 0
total_bits = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = ''.join(map(str, y_test[idx]))
    matches = sum(p == a for p, a in zip(predicted_response, actual_response))
    total_bits += len(actual_response)
    total_matches += matches

# Calculate total accuracy
total_accuracy = calculate_accuracy(total_matches, total_bits)
print(f"Total matches: {total_matches}, Total bits: {total_bits}")
print(f"Overall accuracy: {total_accuracy:.2f}%")


# In[1]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    # Convert DataFrame columns containing binary string data into numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Initialize a model for each bit in the response
    models = [BernoulliNB() for _ in range(response_data.shape[1])]
    # Train each model on the corresponding bit of the response
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    # Predict each bit using the corresponding model
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_1000.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_1000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models on training data
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
total_matches = 0
total_bits = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    matches = np.sum(predicted_response == actual_response)
    total_matches += matches
    total_bits += len(actual_response)

# Calculate total accuracy
accuracy = (total_matches / total_bits) * 100
print(f"Total matches: {total_matches}, Total bits: {total_bits}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[2]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    # Convert DataFrame columns containing binary string data into numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Initialize a model for each bit in the response
    models = [BernoulliNB() for _ in range(response_data.shape[1])]
    # Train each model on the corresponding bit of the response
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenges):
    # Predict each bit using the corresponding model
    predicted_responses = np.column_stack([model.predict(challenges) for model in models])
    return predicted_responses

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_10000.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_10000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models on training data
models = train_models(X_train, y_train)

# Predict responses for test data
predicted_test_responses = predict_response(models, X_test)

# Calculate overall accuracy
accuracy = accuracy_score(y_test.flatten(), predicted_test_responses.flatten()) * 100
print(f"Overall accuracy: {accuracy:.2f}%")


# In[3]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
   # Convert DataFrame columns containing binary string data into numpy arrays of integers
   challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
   response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
   return challenge_data, response_data

def train_models(challenge_data, response_data):
   # Initialize a model for each bit in the response
   models = []
   for _ in range(response_data.shape[1]):
       # You can adjust the class_prior to manage the prior probability if needed
       model = BernoulliNB(class_prior=[0.5, 0.5])  # Example of setting equal priors
       models.append(model)
   # Train each model on the corresponding bit of the response
   for i, model in enumerate(models):
       model.fit(challenge_data, response_data[:, i])
       # Optionally inspect the learned class log probabilities (log of prior probabilities)
       print(f"Model {i} class log probabilities: {model.class_log_prior_}")
   return models

def predict_response(models, challenges):
   # Predict each bit using the corresponding model
   predicted_responses = np.column_stack([model.predict(challenges) for model in models])
   return predicted_responses

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_10000.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_10000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models on training data
models = train_models(X_train, y_train)

# Predict responses for test data
predicted_test_responses = predict_response(models, X_test)

# Calculate overall accuracy
accuracy = accuracy_score(y_test.flatten(), predicted_test_responses.flatten()) * 100
print(f"Overall accuracy: {accuracy:.2f}%")


# In[4]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB  # Using Gaussian Naive Bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    # Convert DataFrame columns containing binary string data into numpy arrays of integers
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Initialize a model for each bit in the response
    models = [GaussianNB() for _ in range(response_data.shape[1])]  # Changed to GaussianNB
    # Train each model on the corresponding bit of the response
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    # Predict each bit using the corresponding model
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_10000.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_10000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models on training data
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
total_matches = 0
total_bits = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    matches = np.sum(predicted_response == actual_response)
    total_matches += matches
    total_bits += len(actual_response)

# Calculate total accuracy
accuracy = (total_matches / total_bits) * 100
print(f"Total matches: {total_matches}, Total bits: {total_bits}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[1]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [GaussianNB() for _ in range(response_data.shape[1])]
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_10000.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_10000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
correct_predictions = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    if np.array_equal(predicted_response, actual_response):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / len(X_test)) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {len(X_test)}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[1]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [GaussianNB() for _ in range(response_data.shape[1])]
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\Set2\\selected_challenges2_10000.txt"
y_text_file = "D:\\data preperation\\New folder\\Set2\\selected_responses2_10000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
correct_predictions = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    if np.array_equal(predicted_response, actual_response):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / len(X_test)) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {len(X_test)}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[2]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [GaussianNB() for _ in range(response_data.shape[1])]
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\Set3\\selected_challenges3_10000.txt"
y_text_file = "D:\\data preperation\\New folder\\Set3\\selected_responses3_10000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
correct_predictions = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    if np.array_equal(predicted_response, actual_response):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / len(X_test)) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {len(X_test)}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[3]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [GaussianNB() for _ in range(response_data.shape[1])]
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\Set3\\selected_challenges3_10000.txt"
y_text_file = "D:\\data preperation\\New folder\\Set3\\selected_responses3_10000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train)

# Collect all predictions
predicted_responses = [predict_response(models, X_test[idx]) for idx in range(len(X_test))]

# Initialize counters
total_matching_bits = 0
total_mismatch_bits = 0
total_bits = 0

# Calculate overall accuracy
for test_challenge, original_response, predicted_response in zip(X_test, y_test, predicted_responses):
    matching_bits = sum(bit1 == bit2 for bit1, bit2 in zip(original_response, predicted_response))
    total_matching_bits += matching_bits
    total_mismatch_bits += len(original_response) - matching_bits
    total_bits += len(original_response)

# Calculate overall accuracy
overall_accuracy = (total_matching_bits / total_bits) * 100

print("Total number of predicted responses:", len(predicted_responses))
print("Total matching bits:", total_matching_bits)
print("Total mismatch bits:", total_mismatch_bits)
print("Total overall bits:", total_bits)
print("Overall accuracy: {:.2f}%".format(overall_accuracy))


# In[4]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [GaussianNB() for _ in range(response_data.shape[1])]
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\Set2\\selected_challenges2_10000.txt"
y_text_file = "D:\\data preperation\\New folder\\Set2\\selected_responses2_10000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train)

# Collect all predictions
predicted_responses = [predict_response(models, X_test[idx]) for idx in range(len(X_test))]

# Initialize counters
total_matching_bits = 0
total_mismatch_bits = 0
total_bits = 0

# Calculate overall accuracy
for test_challenge, original_response, predicted_response in zip(X_test, y_test, predicted_responses):
    matching_bits = sum(bit1 == bit2 for bit1, bit2 in zip(original_response, predicted_response))
    total_matching_bits += matching_bits
    total_mismatch_bits += len(original_response) - matching_bits
    total_bits += len(original_response)

# Calculate overall accuracy
overall_accuracy = (total_matching_bits / total_bits) * 100

print("Total number of predicted responses:", len(predicted_responses))
print("Total matching bits:", total_matching_bits)
print("Total mismatch bits:", total_mismatch_bits)
print("Total overall bits:", total_bits)
print("Overall accuracy: {:.2f}%".format(overall_accuracy))


# In[1]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [GaussianNB() for _ in range(response_data.shape[1])]
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\Set4\\selected_challenges4_10000.txt"
y_text_file = "D:\\data preperation\\New folder\\Set4\\selected_responses4_10000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
correct_predictions = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    if np.array_equal(predicted_response, actual_response):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / len(X_test)) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {len(X_test)}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[2]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [GaussianNB() for _ in range(response_data.shape[1])]
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\Set5\\selected_challenges5_10000.txt"
y_text_file = "D:\\data preperation\\New folder\\Set5\\selected_responses5_10000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
correct_predictions = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    if np.array_equal(predicted_response, actual_response):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / len(X_test)) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {len(X_test)}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[3]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [GaussianNB() for _ in range(response_data.shape[1])]
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return ''.join(map(str, predicted_response))

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\sampled_challenges_1000.txt"
y_text_file = "D:\\data preperation\\New folder\\sampled_responses_1000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train)

# Collecting data for CSV output
predicted_responses = []
actual_responses = []

# Evaluate the model on each test challenge and store results for CSV
correct_predictions = 0
for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = ''.join(map(str, y_test[idx]))
    predicted_responses.append(predicted_response)
    actual_responses.append(actual_response)
    if predicted_response == actual_response:
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / len(X_test)) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {len(X_test)}")
print(f"Overall accuracy: {accuracy:.2f}%")

# Save predicted and actual responses to separate text files
with open("D:\\data preperation\\New folder\\predicted_responses.txt", "w") as f_pred:
    for response in predicted_responses:
        f_pred.write(response + "\n")

with open("D:\\data preperation\\New folder\\actual_responses.txt", "w") as f_actual:
    for response in actual_responses:
        f_actual.write(response + "\n")

print("Predicted and actual responses saved successfully in .txt files.")


# In[4]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [GaussianNB() for _ in range(response_data.shape[1])]
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return ''.join(map(str, predicted_response))

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\sampled_challenges_1000.txt"
y_text_file = "D:\\data preperation\\New folder\\sampled_responses_1000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
predicted_responses = []
actual_responses = []
correct_bit_predictions = 0
total_bits = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = ''.join(map(str, y_test[idx]))
    predicted_responses.append(predicted_response)
    actual_responses.append(actual_response)
    # Count positionally correct bits
    for pred_bit, act_bit in zip(predicted_response, actual_response):
        if pred_bit == act_bit:
            correct_bit_predictions += 1
    total_bits += len(actual_response)

# Calculate the bit-wise accuracy
accuracy = (correct_bit_predictions / total_bits) * 100 if total_bits > 0 else 0
print(f"Total correct bit predictions: {correct_bit_predictions}, Total bits: {total_bits}")
print(f"Bit-wise accuracy: {accuracy:.2f}%")

# Save predicted and actual responses to separate text files
with open("D:\\data preperation\\New folder\\predicted_responses.txt", "w") as f_pred:
    for response in predicted_responses:
        f_pred.write(response + "\n")

with open("D:\\data preperation\\New folder\\actual_responses.txt", "w") as f_actual:
    for response in actual_responses:
        f_actual.write(response + "\n")

print("Predicted and actual responses saved successfully in .txt files.")


# In[5]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [GaussianNB() for _ in range(response_data.shape[1])]
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\sampled_challenges_1000.txt"
y_text_file = "D:\\data preperation\\New folder\\sampled_responses_1000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
correct_predictions = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    if np.array_equal(predicted_response, actual_response):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / len(X_test)) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {len(X_test)}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[1]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [GaussianNB() for _ in range(response_data.shape[1])]
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\sampled_challenges_2000.txt"
y_text_file = "D:\\data preperation\\New folder\\sampled_responses_2000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
correct_predictions = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    if np.array_equal(predicted_response, actual_response):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / len(X_test)) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {len(X_test)}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[2]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [GaussianNB() for _ in range(response_data.shape[1])]
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\sampled_challenges_5000.txt"
y_text_file = "D:\\data preperation\\New folder\\sampled_responses_5000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
correct_predictions = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    if np.array_equal(predicted_response, actual_response):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / len(X_test)) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {len(X_test)}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[3]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [GaussianNB() for _ in range(response_data.shape[1])]
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\sampled_challenges_8000.txt"
y_text_file = "D:\\data preperation\\New folder\\sampled_responses_8000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
correct_predictions = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    if np.array_equal(predicted_response, actual_response):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / len(X_test)) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {len(X_test)}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[4]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [GaussianNB() for _ in range(response_data.shape[1])]
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\sampled_challenges_5000.txt"
y_text_file = "D:\\data preperation\\New folder\\sampled_responses_5000.txt"
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

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    if np.array_equal(predicted_response, actual_response):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / len(X_test)) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {len(X_test)}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[1]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [GaussianNB() for _ in range(response_data.shape[1])]
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\last_10000_challenges.txt"
y_text_file = "D:\\data preperation\\New folder\\last_10000_responses.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
correct_predictions = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    if np.array_equal(predicted_response, actual_response):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / len(X_test)) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {len(X_test)}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[1]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [GaussianNB() for _ in range(response_data.shape[1])]
    for i, model in enumerate(models):
        model.fit(challenge_data, response_data[:, i])
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = np.array([model.predict(new_challenge_array) for model in models]).flatten()
    return predicted_response

# Load challenge and response data from files
x_text_file = "E:\\Study Oulu\\2. Second period\\Hardware Hacing\\Challenge.txt"
y_text_file = "E:\\Study Oulu\\2. Second period\\Hardware Hacing\\Response.txt



E:\\Study Oulu\\2. Second period\\Hardware Hacing\\Response.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train)

# Evaluate the model on each test challenge
correct_predictions = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    if np.array_equal(predicted_response, actual_response):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / len(X_test)) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {len(X_test)}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[ ]:




