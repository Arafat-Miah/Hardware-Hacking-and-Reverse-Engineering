#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Changed import here
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
from sklearn.ensemble import RandomForestClassifier  # Changed import here
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
from sklearn.ensemble import RandomForestClassifier  # Changed import here
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


# In[1]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Changed import here
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


# In[1]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Changed import here
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


# In[1]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = [RandomForestClassifier(n_estimators=100) for _ in range(response_data.shape[1])]
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
with open("D:\\data preperation\\New folder\\predicted_responses_rf.txt", "w") as f_pred:
    for response in predicted_responses:
        f_pred.write(response + "\n")

with open("D:\\data preperation\\New folder\\actual_responses_rf.txt", "w") as f_actual:
    for response in actual_responses:
        f_actual.write(response + "\n")

print("Predicted and actual responses saved successfully in .txt files.")


# In[2]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Changed import here
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
from sklearn.ensemble import RandomForestClassifier  # Changed import here
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
from sklearn.ensemble import RandomForestClassifier  # Changed import here
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
from sklearn.ensemble import RandomForestClassifier  # Changed import here
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


# In[1]:


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


# In[8]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Changed import here
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


# In[7]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Changed import here
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
x_text_file = "D:\\data preperation\\New folder\\challenge_long.txt"
y_text_file = "D:\\data preperation\\New folder\\response_long.txt"
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


# In[ ]:




