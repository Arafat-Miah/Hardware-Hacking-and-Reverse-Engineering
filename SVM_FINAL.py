#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd

# Load the dataset from the first text file (x-axis data)
x_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\input10.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'], sep='\t')

# Load the dataset from the second text file (y-axis data)
y_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\output.txt"
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Convert the binary strings in x_df to binary data
x_df['Challenge'] = x_df['Challenge'].apply(lambda x: int(x, 2) if isinstance(x, str) else x)

# Concatenate x_df and y_df along the columns axis
combined_df = pd.concat([x_df, y_df], axis=1)

# Print the combined DataFrame
print("Combined DataFrame:")
print(combined_df)


# In[19]:


import pandas as pd

# Load the dataset from the first text file (x-axis data) as strings
x_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\input10.txt"
with open(x_text_file, 'r') as file:
    lines = file.readlines()
x_df = pd.DataFrame({'Challenge': lines})

# Load the dataset from the second text file (y-axis data)
y_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\output.txt"
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Concatenate x_df and y_df along the columns axis
combined_df = pd.concat([x_df, y_df], axis=1)

# Print the combined DataFrame
print("Combined DataFrame:")
print(combined_df)


# In[22]:


import pandas as pd

# Load the dataset from the first text file (x-axis data) as strings
x_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\input10.txt"
with open(x_text_file, 'r') as file:
    lines = [line.strip() for line in file.readlines()]  # Remove newline characters
x_df = pd.DataFrame({'Challenge': lines})

# Load the dataset from the second text file (y-axis data)
y_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\output.txt"
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Concatenate x_df and y_df along the columns axis
combined_df = pd.concat([x_df, y_df], axis=1)

# Print the combined DataFrame
print("Combined DataFrame:")
print(combined_df)
# User input for the binary challenge
user_challenge = 0b000000000000000000000111  # User's challenge in binary format

# Find the index of the user_challenge in the "Challenge" column of x_df
challenge_index = x_df.index[x_df["Challenge"] == f"{user_challenge:024b}"].tolist()

if challenge_index:
    # Get the corresponding value from y_df
    corresponding_y_value = y_df.iloc[challenge_index[0]]['Y_Data']
    print("Index of Challenge", format(user_challenge, '024b'), ":", challenge_index[0])
    print("Corresponding Y value:", corresponding_y_value)
else:
    print("No corresponding Challenge found in the dataset.")


# In[26]:


import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset from the first text file (x-axis data) as strings
x_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\input10.txt"
with open(x_text_file, 'r') as file:
    lines = [line.strip() for line in file.readlines()]  # Remove newline characters
x_df = pd.DataFrame({'Challenge': lines})

# Load the dataset from the second text file (y-axis data)
y_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\output.txt"
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Concatenate x_df and y_df along the columns axis
combined_df = pd.concat([x_df, y_df], axis=1)

# Print the combined DataFrame
print("Combined DataFrame:")
print(combined_df)

# Splitting the data into training and testing sets
X = combined_df['Challenge']
y = combined_df['Y_Data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input data to 2D array
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

# Initialize the SVM classifier with RBF kernel
svm_classifier = SVC(kernel='rbf')

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[33]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset from the first text file (x-axis data) as strings
x_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\input10.txt"
with open(x_text_file, 'r') as file:
    lines = [line.strip() for line in file.readlines()]  # Remove newline characters
x_df = pd.DataFrame({'Challenge': lines})

# Load the dataset from the second text file (y-axis data)
y_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\output.txt"
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Concatenate x_df and y_df along the columns axis
combined_df = pd.concat([x_df, y_df], axis=1)

# Splitting the data into training and testing sets (80% training, 20% testing)
X = combined_df['Challenge']
y = combined_df['Y_Data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the binary strings to arrays of integers
X_train = X_train.apply(lambda x: [int(digit) for digit in x])
X_test = X_test.apply(lambda x: [int(digit) for digit in x])

# Reshape the data into a 2D array
X_train = X_train.values.tolist()
X_test = X_test.values.tolist()

# Initialize the SVM classifier with RBF kernel
svm_classifier = SVC(kernel='rbf')

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# User-provided challenge in binary
user_challenge_binary = "000000000000000000001001"

# Convert the user challenge to a list of integers and reshape it into a 2D array
user_challenge_input = [[int(digit) for digit in user_challenge_binary]]

# Predict the response for the user challenge
user_challenge_response = svm_classifier.predict(user_challenge_input)

print("Predicted response for the user challenge:", user_challenge_response)


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset from the first text file (x-axis data) as strings
x_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\input10.txt"
with open(x_text_file, 'r') as file:
    lines = [line.strip() for line in file.readlines()]  # Remove newline characters
x_df = pd.DataFrame({'Challenge': lines})

# Load the dataset from the second text file (y-axis data)
y_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\output.txt"
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Concatenate x_df and y_df along the columns axis
combined_df = pd.concat([x_df, y_df], axis=1)

# Splitting the data into training and testing sets (80% training, 20% testing)
X = combined_df['Challenge']
y = combined_df['Y_Data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the binary strings to arrays of integers
X_train = X_train.apply(lambda x: [int(digit) for digit in x])
X_test = X_test.apply(lambda x: [int(digit) for digit in x])

# Reshape the data into a 2D array
X_train = X_train.values.tolist()
X_test = X_test.values.tolist()

# Initialize the SVM classifier with RBF kernel
svm_classifier = SVC(kernel='rbf')

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# User-provided challenge in binary
user_challenge_binary = "000000000000000000001000"

# Convert the user challenge to a list of integers and reshape it into a 2D array
user_challenge_input = [[int(digit) for digit in user_challenge_binary]]

# Predict the response for the user challenge
user_challenge_response = svm_classifier.predict(user_challenge_input)

# Retrieve the original response corresponding to the user's challenge
original_response = combined_df.loc[combined_df['Challenge'] == user_challenge_binary, 'Y_Data'].values[0]

print("Predicted response for the user challenge:", user_challenge_response)
print("Original response for the user challenge:", original_response)


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset from the first text file (x-axis data) as strings
x_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\input10.txt"
with open(x_text_file, 'r') as file:
    lines = [line.strip() for line in file.readlines()]  # Remove newline characters
x_df = pd.DataFrame({'Challenge': lines})

# Load the dataset from the second text file (y-axis data)
y_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\output.txt"
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Concatenate x_df and y_df along the columns axis
combined_df = pd.concat([x_df, y_df], axis=1)

# Splitting the data into training and testing sets (80% training, 20% testing)
X = combined_df['Challenge']
y = combined_df['Y_Data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the binary strings to arrays of integers
X_train = X_train.apply(lambda x: [int(digit) for digit in x])
X_test = X_test.apply(lambda x: [int(digit) for digit in x])

# Reshape the data into a 2D array
X_train = X_train.values.tolist()
X_test = X_test.values.tolist()

# Initialize the SVM classifier with RBF kernel
svm_classifier = SVC(kernel='rbf')

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm_classifier.predict(X_test)

# User-provided challenge in binary
user_challenge_binary = "000000000000000000001000"

# Convert the user challenge to a list of integers and reshape it into a 2D array
user_challenge_input = [[int(digit) for digit in user_challenge_binary]]

# Predict the response for the user challenge
user_challenge_response = svm_classifier.predict(user_challenge_input)

# Retrieve the original response corresponding to the user's challenge
original_response = combined_df.loc[combined_df['Challenge'] == user_challenge_binary, 'Y_Data'].values[0]

print("Predicted response for the user challenge:", user_challenge_response)
print("Original response for the user challenge:", original_response)

# Define a function to calculate accuracy between two binary strings
def calculate_accuracy(binary_str1, binary_str2):
    if len(binary_str1) != len(binary_str2):
        raise ValueError("Binary strings must have the same length")
    
    total_bits = len(binary_str1)
    matching_bits = sum(bit1 == bit2 for bit1, bit2 in zip(binary_str1, binary_str2))
    
    accuracy = (matching_bits / total_bits) * 100
    return accuracy

# Calculate accuracy
accuracy = calculate_accuracy(user_challenge_response[0], original_response)
print("Accuracy between the two binary strings: {:.2f}%".format(accuracy))


# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset from the first text file (x-axis data) as strings
x_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\input10.txt"
with open(x_text_file, 'r') as file:
    lines = [line.strip() for line in file.readlines()]  # Remove newline characters
x_df = pd.DataFrame({'Challenge': lines})

# Load the dataset from the second text file (y-axis data)
y_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\output.txt"
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Concatenate x_df and y_df along the columns axis
combined_df = pd.concat([x_df, y_df], axis=1)

# Splitting the data into training and testing sets (80% training, 20% testing)
X = combined_df['Challenge']
y = combined_df['Y_Data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the binary strings to arrays of integers
X_train = X_train.apply(lambda x: [int(digit) for digit in x])
X_test = X_test.apply(lambda x: [int(digit) for digit in x])

# Reshape the data into a 2D array
X_train = X_train.values.tolist()
X_test = X_test.values.tolist()

# Initialize the SVM classifier with RBF kernel
svm_classifier = SVC(kernel='rbf')

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm_classifier.predict(X_test)

# User-provided challenge in binary
user_challenge_binary = "000000000000000000001000"

# Convert the user challenge to a list of integers and reshape it into a 2D array
user_challenge_input = [[int(digit) for digit in user_challenge_binary]]

# Predict the response for the user challenge
user_challenge_response = svm_classifier.predict(user_challenge_input)

# Retrieve the original response corresponding to the user's challenge
original_response = combined_df.loc[combined_df['Challenge'] == user_challenge_binary, 'Y_Data'].values[0]

print("Predicted response for the user challenge:", user_challenge_response)
print("Original response for the user challenge:", original_response)

# Define a function to calculate accuracy between two binary strings
def calculate_accuracy(binary_str1, binary_str2):
    if len(binary_str1) != len(binary_str2):
        raise ValueError("Binary strings must have the same length")
    
    total_bits = len(binary_str1)
    matching_bits = sum(bit1 == bit2 for bit1, bit2 in zip(binary_str1, binary_str2))
    mismatching_bits = total_bits - matching_bits
    
    accuracy = (matching_bits / total_bits) * 100
    return accuracy, matching_bits, mismatching_bits

# Calculate accuracy and count matching/mismatching bits
accuracy, matching_bits, mismatching_bits = calculate_accuracy(user_challenge_response[0], original_response)
print("Accuracy between the two binary strings: {:.2f}%".format(accuracy))
print("Number of matching bits:", matching_bits)
print("Number of mismatching bits:", mismatching_bits)


# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset from the first text file (x-axis data) as strings
x_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\input10.txt"
with open(x_text_file, 'r') as file:
    lines = [line.strip() for line in file.readlines()]  # Remove newline characters
x_df = pd.DataFrame({'Challenge': lines})

# Load the dataset from the second text file (y-axis data)
y_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\output.txt"
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Concatenate x_df and y_df along the columns axis
combined_df = pd.concat([x_df, y_df], axis=1)

# Splitting the data into training and testing sets (80% training, 20% testing)
X = combined_df['Challenge']
y = combined_df['Y_Data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the binary strings to arrays of integers
X_train = X_train.apply(lambda x: [int(digit) for digit in x])
X_test = X_test.apply(lambda x: [int(digit) for digit in x])

# Reshape the data into a 2D array
X_train = X_train.values.tolist()
X_test = X_test.values.tolist()

# Initialize the SVM classifier with RBF kernel
svm_classifier = SVC(kernel='rbf')

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Predict responses for all test challenges
predicted_responses = svm_classifier.predict(X_test)

# Calculate overall accuracy
total_matching_bits = 0
total_bits = 0

for test_challenge, original_response, predicted_response in zip(X_test, y_test, predicted_responses):
    # Convert test_challenge and original_response to strings
    test_challenge_str = ''.join(map(str, test_challenge))
    original_response_str = str(original_response)
    
    # Calculate accuracy for each predicted response
    matching_bits = sum(bit1 == bit2 for bit1, bit2 in zip(original_response_str, predicted_response))
    total_matching_bits += matching_bits
    total_bits += len(original_response_str)

# Calculate overall accuracy
overall_accuracy = (total_matching_bits / total_bits) * 100

print("Overall accuracy:", overall_accuracy)


# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset from the first text file (x-axis data) as strings
x_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\input10.txt"
with open(x_text_file, 'r') as file:
    lines = [line.strip() for line in file.readlines()]  # Remove newline characters
x_df = pd.DataFrame({'Challenge': lines})

# Load the dataset from the second text file (y-axis data)
y_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\output.txt"
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Concatenate x_df and y_df along the columns axis
combined_df = pd.concat([x_df, y_df], axis=1)

# Splitting the data into training and testing sets (80% training, 20% testing)
X = combined_df['Challenge']
y = combined_df['Y_Data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the binary strings to arrays of integers
X_train = X_train.apply(lambda x: [int(digit) for digit in x])
X_test = X_test.apply(lambda x: [int(digit) for digit in x])

# Reshape the data into a 2D array
X_train = X_train.values.tolist()
X_test = X_test.values.tolist()

# Initialize the SVM classifier with RBF kernel
svm_classifier = SVC(kernel='rbf')

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Predict responses for all test challenges
predicted_responses = svm_classifier.predict(X_test)

# Initialize counters
total_predicted_responses = len(predicted_responses)
total_matching_bits = 0
total_mismatch_bits = 0
total_bits = 0

# Calculate overall accuracy
for test_challenge, original_response, predicted_response in zip(X_test, y_test, predicted_responses):
    # Convert test_challenge and original_response to strings
    test_challenge_str = ''.join(map(str, test_challenge))
    original_response_str = str(original_response)
    
    # Calculate accuracy for each predicted response
    matching_bits = sum(bit1 == bit2 for bit1, bit2 in zip(original_response_str, predicted_response))
    total_matching_bits += matching_bits
    total_mismatch_bits += len(original_response_str) - matching_bits
    total_bits += len(original_response_str)

# Calculate overall accuracy
overall_accuracy = (total_matching_bits / total_bits) * 100

print("Total number of predicted responses:", total_predicted_responses)
print("Total matching bits:", total_matching_bits)
print("Total mismatch bits:", total_mismatch_bits)
print("Total overall bits:", total_bits)
print("Overall accuracy:", overall_accuracy)


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset from the first text file (x-axis data) as strings
x_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\selected_challenges.txt"
with open(x_text_file, 'r') as file:
    lines = [line.strip() for line in file.readlines()]  # Remove newline characters
x_df = pd.DataFrame({'Challenge': lines})

# Load the dataset from the second text file (y-axis data)
y_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\selected_responses.txt"
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Concatenate x_df and y_df along the columns axis
combined_df = pd.concat([x_df, y_df], axis=1)

# Splitting the data into training and testing sets (80% training, 20% testing)
X = combined_df['Challenge']
y = combined_df['Y_Data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the binary strings to arrays of integers
X_train = X_train.apply(lambda x: [int(digit) for digit in x])
X_test = X_test.apply(lambda x: [int(digit) for digit in x])

# Reshape the data into a 2D array
X_train = X_train.values.tolist()
X_test = X_test.values.tolist()

# Initialize the SVM classifier with RBF kernel
svm_classifier = SVC(kernel='rbf')

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Predict responses for all test challenges
predicted_responses = svm_classifier.predict(X_test)

# Initialize counters
total_predicted_responses = len(predicted_responses)
total_matching_bits = 0
total_mismatch_bits = 0
total_bits = 0

# Calculate overall accuracy
for test_challenge, original_response, predicted_response in zip(X_test, y_test, predicted_responses):
    # Convert test_challenge and original_response to strings
    test_challenge_str = ''.join(map(str, test_challenge))
    original_response_str = str(original_response)
    
    # Calculate accuracy for each predicted response
    matching_bits = sum(bit1 == bit2 for bit1, bit2 in zip(original_response_str, predicted_response))
    total_matching_bits += matching_bits
    total_mismatch_bits += len(original_response_str) - matching_bits
    total_bits += len(original_response_str)

# Calculate overall accuracy
overall_accuracy = (total_matching_bits / total_bits) * 100

print("Total number of predicted responses:", total_predicted_responses)
print("Total matching bits:", total_matching_bits)
print("Total mismatch bits:", total_mismatch_bits)
print("Total overall bits:", total_bits)
print("Overall accuracy:", overall_accuracy)


# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset from the first text file (x-axis data) as strings
x_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\selected_challenges_1000.txt"
with open(x_text_file, 'r') as file:
    lines = [line.strip() for line in file.readlines()]  # Remove newline characters
x_df = pd.DataFrame({'Challenge': lines})

# Load the dataset from the second text file (y-axis data)
y_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\selected_responses_1000.txt"
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Concatenate x_df and y_df along the columns axis
combined_df = pd.concat([x_df, y_df], axis=1)

# Splitting the data into training and testing sets (80% training, 20% testing)
X = combined_df['Challenge']
y = combined_df['Y_Data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the binary strings to arrays of integers
X_train = X_train.apply(lambda x: [int(digit) for digit in x])
X_test = X_test.apply(lambda x: [int(digit) for digit in x])

# Reshape the data into a 2D array
X_train = X_train.values.tolist()
X_test = X_test.values.tolist()

# Initialize the SVM classifier with RBF kernel
svm_classifier = SVC(kernel='rbf')

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Predict responses for all test challenges
predicted_responses = svm_classifier.predict(X_test)

# Initialize counters
total_predicted_responses = len(predicted_responses)
total_matching_bits = 0
total_mismatch_bits = 0
total_bits = 0

# Calculate overall accuracy
for test_challenge, original_response, predicted_response in zip(X_test, y_test, predicted_responses):
    # Convert test_challenge and original_response to strings
    test_challenge_str = ''.join(map(str, test_challenge))
    original_response_str = str(original_response)
    
    # Calculate accuracy for each predicted response
    matching_bits = sum(bit1 == bit2 for bit1, bit2 in zip(original_response_str, predicted_response))
    total_matching_bits += matching_bits
    total_mismatch_bits += len(original_response_str) - matching_bits
    total_bits += len(original_response_str)

# Calculate overall accuracy
overall_accuracy = (total_matching_bits / total_bits) * 100

print("Total number of predicted responses:", total_predicted_responses)
print("Total matching bits:", total_matching_bits)
print("Total mismatch bits:", total_mismatch_bits)
print("Total overall bits:", total_bits)
print("Overall accuracy:", overall_accuracy)


# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset from the first text file (x-axis data) as strings
x_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\challenges_1000_ara_fai.txt"
with open(x_text_file, 'r') as file:
    lines = [line.strip() for line in file.readlines()]  # Remove newline characters
x_df = pd.DataFrame({'Challenge': lines})

# Load the dataset from the second text file (y-axis data)
y_text_file = "C:\\Users\\Asus\\Desktop\\all data\\2 lac data\\response_1000_from_ara_fai.txt"
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Concatenate x_df and y_df along the columns axis
combined_df = pd.concat([x_df, y_df], axis=1)

# Splitting the data into training and testing sets (80% training, 20% testing)
X = combined_df['Challenge']
y = combined_df['Y_Data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the binary strings to arrays of integers
X_train = X_train.apply(lambda x: [int(digit) for digit in x])
X_test = X_test.apply(lambda x: [int(digit) for digit in x])

# Reshape the data into a 2D array
X_train = X_train.values.tolist()
X_test = X_test.values.tolist()

# Initialize the SVM classifier with RBF kernel
svm_classifier = SVC(kernel='rbf')

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Predict responses for all test challenges
predicted_responses = svm_classifier.predict(X_test)

# Initialize counters
total_predicted_responses = len(predicted_responses)
total_matching_bits = 0
total_mismatch_bits = 0
total_bits = 0

# Calculate overall accuracy
for test_challenge, original_response, predicted_response in zip(X_test, y_test, predicted_responses):
    # Convert test_challenge and original_response to strings
    test_challenge_str = ''.join(map(str, test_challenge))
    original_response_str = str(original_response)
    
    # Calculate accuracy for each predicted response
    matching_bits = sum(bit1 == bit2 for bit1, bit2 in zip(original_response_str, predicted_response))
    total_matching_bits += matching_bits
    total_mismatch_bits += len(original_response_str) - matching_bits
    total_bits += len(original_response_str)

# Calculate overall accuracy
overall_accuracy = (total_matching_bits / total_bits) * 100

print("Total number of predicted responses:", total_predicted_responses)
print("Total matching bits:", total_matching_bits)
print("Total mismatch bits:", total_mismatch_bits)
print("Total overall bits:", total_bits)
print("Overall accuracy:", overall_accuracy)


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset from the first text file (x-axis data) as strings
x_text_file = "D:\\data preperation\\selected_challenges_10000.txt"
with open(x_text_file, 'r') as file:
    lines = [line.strip() for line in file.readlines()]  # Remove newline characters
x_df = pd.DataFrame({'Challenge': lines})

# Load the dataset from the second text file (y-axis data)
y_text_file = "D:\\data preperation\\selected_responses_10000.txt"
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Concatenate x_df and y_df along the columns axis
combined_df = pd.concat([x_df, y_df], axis=1)

# Splitting the data into training and testing sets (80% training, 20% testing)
X = combined_df['Challenge']
y = combined_df['Y_Data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the binary strings to arrays of integers
X_train = X_train.apply(lambda x: [int(digit) for digit in x])
X_test = X_test.apply(lambda x: [int(digit) for digit in x])

# Reshape the data into a 2D array
X_train = X_train.values.tolist()
X_test = X_test.values.tolist()

# Initialize the SVM classifier with RBF kernel
svm_classifier = SVC(kernel='rbf')

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Predict responses for all test challenges
predicted_responses = svm_classifier.predict(X_test)

# Initialize counters
total_predicted_responses = len(predicted_responses)
total_matching_bits = 0
total_mismatch_bits = 0
total_bits = 0

# Calculate overall accuracy
for test_challenge, original_response, predicted_response in zip(X_test, y_test, predicted_responses):
    # Convert test_challenge and original_response to strings
    test_challenge_str = ''.join(map(str, test_challenge))
    original_response_str = str(original_response)
    
    # Calculate accuracy for each predicted response
    matching_bits = sum(bit1 == bit2 for bit1, bit2 in zip(original_response_str, predicted_response))
    total_matching_bits += matching_bits
    total_mismatch_bits += len(original_response_str) - matching_bits
    total_bits += len(original_response_str)

# Calculate overall accuracy
overall_accuracy = (total_matching_bits / total_bits) * 100

print("Total number of predicted responses:", total_predicted_responses)
print("Total matching bits:", total_matching_bits)
print("Total mismatch bits:", total_mismatch_bits)
print("Total overall bits:", total_bits)
print("Overall accuracy:", overall_accuracy)


# In[2]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    # Using SVM with a linear kernel
    models = [SVC(kernel='linear') for _ in range(response_data.shape[1])]
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


# In[4]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def check_class_distribution(response_data):
    valid_indices = []
    for i in range(response_data.shape[1]):
        unique_classes = np.unique(response_data[:, i])
        if len(unique_classes) < 2:
            print(f"Column {i} has less than two classes: {unique_classes}")
        else:
            valid_indices.append(i)
            print(f"Column {i} classes: {unique_classes}")
    return valid_indices

def train_models(challenge_data, response_data, valid_indices):
    models = [None] * response_data.shape[1]  # Initialize a list of models for each column
    for i in valid_indices:
        model = SVC(kernel='linear')
        model.fit(challenge_data, response_data[:, i])
        models[i] = model
    return models

def predict_response(models, challenge):
    predicted_response = []
    for model in models:
        if model is not None:
            new_challenge_array = np.array([challenge])
            predicted = model.predict(new_challenge_array)[0]
        else:
            predicted = None  # Default or assumed prediction for unsupported models
        predicted_response.append(predicted)
    return predicted_response

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_10000.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_10000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Check class distribution and get valid indices for training
valid_indices = check_class_distribution(response_data)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data[:, valid_indices], test_size=0.2, random_state=42, shuffle=True)

# Train models
models = train_models(X_train, y_train, valid_indices)

# Evaluate the model on each test challenge
correct_predictions = 0
total_tests = len(X_test)

for idx in range(total_tests):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx, :]
    # Count correct predictions (handling None for unsupported models)
    correct_predictions += np.sum([(p == a) if p is not None else False for p, a in zip(predicted_response, actual_response)])

# Calculate total accuracy
accuracy = (correct_predictions / (total_tests * len(valid_indices))) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests * len(valid_indices)}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[11]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Check if there are at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            print(f"Skipping column {i} due to having less than two classes.")
            models.append(None)  # Append None for models that were not trained
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:  # Only predict if a model was trained
            predicted_response.append(model.predict(new_challenge_array)[0])
        else:
            predicted_response.append(None)  # Handle skipped models
    return np.array(predicted_response)

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\selected_challenges_1000.txt"
y_text_file = "D:\\data preperation\\New folder\\selected_responses_1000.txt"
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
total_predictions = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Increment total predictions and correct predictions only for valid predictions
    for pred, act in zip(predicted_response, actual_response):
        if pred is not None:  # Check if prediction was made
            total_predictions += 1
            if pred == act:
                correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
print(f"Total correct predictions: {correct_predictions}, Total valid tests: {total_predictions}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[12]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

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
total_tests = len(X_test)

for idx in range(total_tests):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Check if the complete responses match, ignoring None predictions
    if np.array_equal(predicted_response[predicted_response != np.array(None)], 
                      actual_response[predicted_response != np.array(None)]):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_tests) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[1]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

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
total_tests = len(X_test)

for idx in range(total_tests):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Check if the complete responses match, ignoring None predictions
    if np.array_equal(predicted_response[predicted_response != np.array(None)], 
                      actual_response[predicted_response != np.array(None)]):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_tests) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[2]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

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
total_tests = len(X_test)

for idx in range(total_tests):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Check if the complete responses match, ignoring None predictions
    if np.array_equal(predicted_response[predicted_response != np.array(None)], 
                      actual_response[predicted_response != np.array(None)]):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_tests) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[1]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

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
total_tests = len(X_test)

for idx in range(total_tests):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Check if the complete responses match, ignoring None predictions
    if np.array_equal(predicted_response[predicted_response != np.array(None)], 
                      actual_response[predicted_response != np.array(None)]):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_tests) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[2]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

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
total_tests = len(X_test)

for idx in range(total_tests):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Check if the complete responses match, ignoring None predictions
    if np.array_equal(predicted_response[predicted_response != np.array(None)], 
                      actual_response[predicted_response != np.array(None)]):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_tests) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[1]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

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
total_tests = len(X_test)

for idx in range(total_tests):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Check if the complete responses match, ignoring None predictions
    if np.array_equal(predicted_response[predicted_response != np.array(None)], 
                      actual_response[predicted_response != np.array(None)]):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_tests) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[2]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

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

# Initialize counters for accuracy calculations
correct_bit_predictions = 0
total_bits = len(X_test) * response_data.shape[1]  # Total bits is the number of test cases times the number of bits per case

# Evaluate the model on each test challenge
for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Count positionally correct bits
    for bit_idx in range(len(predicted_response)):
        if predicted_response[bit_idx] == actual_response[bit_idx]:
            correct_bit_predictions += 1

# Calculate the bit-wise accuracy
accuracy = (correct_bit_predictions / total_bits) * 100
print(f"Total correct bit predictions: {correct_bit_predictions}, Total bits: {total_bits}")
print(f"Bit-wise accuracy: {accuracy:.2f}%")


# In[3]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

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

# Initialize counters for accuracy calculations
correct_bit_predictions = 0
total_bits = len(X_test) * response_data.shape[1]  # Total bits is the number of test cases times the number of bits per case

# Evaluate the model on each test challenge
for idx in range(len(X_test)):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Count positionally correct bits
    for bit_idx in range(len(predicted_response)):
        if predicted_response[bit_idx] == actual_response[bit_idx]:
            correct_bit_predictions += 1

# Calculate the bit-wise accuracy
accuracy = (correct_bit_predictions / total_bits) * 100
print(f"Total correct bit predictions: {correct_bit_predictions}, Total bits: {total_bits}")
print(f"Bit-wise accuracy: {accuracy:.2f}%")


# In[6]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    default_values = []
    for i in range(response_data.shape[1]):
        unique_values = np.unique(response_data[:, i])
        if len(unique_values) > 1:
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
            default_values.append(None)  # No default needed where model is trained
        else:
            models.append(None)  # No model for columns with insufficient class diversity
            default_values.append(str(unique_values[0]))  # Default to the only value present
    return models, default_values

def predict_response(models, default_values, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model, default in zip(models, default_values):
        if model is not None:
            predicted_response.append(str(int(model.predict(new_challenge_array)[0])))
        else:
            predicted_response.append(default)  # Use the default value if no model was trained
    return ''.join(predicted_response)

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\sampled_challenges_2000.txt"
y_text_file = "D:\\data preperation\\New folder\\sampled_responses_2000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models and get default values for untrainable bits
models, default_values = train_models(X_train, y_train)

# Evaluate the model on each test challenge
predicted_responses = []
actual_responses = []
correct_bit_predictions = 0
total_bits = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, default_values, X_test[idx])
    actual_response = ''.join(map(str, y_test[idx]))
    predicted_responses.append(predicted_response)
    actual_responses.append(actual_response)
    # Count positionally correct bits
    for pred_bit, act_bit in zip(predicted_response, actual_response):
        if pred_bit == act_bit:
            correct_bit_predictions += 1
    total_bits += len(actual_response)

# Save predicted and actual responses to files
predicted_file_path = "D:\\data preperation\\New folder\\predicted_responses_svm.txt"
actual_file_path = "D:\\data preperation\\New folder\\actual_responses_svm.txt"

with open(predicted_file_path, 'w') as file:
    for response in predicted_responses:
        file.write(response + '\n')

with open(actual_file_path, 'w') as file:
    for response in actual_responses:
        file.write(response + '\n')

# Calculate the bit-wise accuracy
accuracy = (correct_bit_predictions / total_bits) * 100 if total_bits > 0 else 0
print(f"Total correct bit predictions: {correct_bit_predictions}, Total bits: {total_bits}")
print(f"Bit-wise accuracy: {accuracy:.2f}%")
print(f"Predicted responses saved to '{predicted_file_path}'.")
print(f"Actual responses saved to '{actual_file_path}'.")


# In[3]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    default_values = []
    for i in range(response_data.shape[1]):
        unique_values = np.unique(response_data[:, i])
        if len(unique_values) > 1:
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
            default_values.append(None)  # No default needed where model is trained
        else:
            models.append(None)  # No model for columns with insufficient class diversity
            default_values.append(str(unique_values[0]))  # Default to the only value present
    return models, default_values

def predict_response(models, default_values, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model, default in zip(models, default_values):
        if model is not None:
            predicted_response.append(str(int(model.predict(new_challenge_array)[0])))
        else:
            predicted_response.append(default)  # Use the default value if no model was trained
    return ''.join(predicted_response)

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\sampled_challenges_5000.txt"
y_text_file = "D:\\data preperation\\New folder\\sampled_responses_5000.txt"
x_df = pd.read_csv(x_text_file, header=None, names=['Challenge'])
y_df = pd.read_csv(y_text_file, header=None, names=['Y_Data'])

# Prepare data
challenge_data, response_data = prepare_data(x_df, y_df)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(challenge_data, response_data, test_size=0.2, random_state=42, shuffle=True)

# Train models and get default values for untrainable bits
models, default_values = train_models(X_train, y_train)

# Evaluate the model on each test challenge
predicted_responses = []
actual_responses = []
correct_bit_predictions = 0
total_bits = 0

for idx in range(len(X_test)):
    predicted_response = predict_response(models, default_values, X_test[idx])
    actual_response = ''.join(map(str, y_test[idx]))
    predicted_responses.append(predicted_response)
    actual_responses.append(actual_response)
    # Count positionally correct bits
    for pred_bit, act_bit in zip(predicted_response, actual_response):
        if pred_bit == act_bit:
            correct_bit_predictions += 1
    total_bits += len(actual_response)

# Save predicted and actual responses to files
predicted_file_path = "D:\\data preperation\\New folder\\predicted_responses_svm.txt"
actual_file_path = "D:\\data preperation\\New folder\\actual_responses_svm.txt"

with open(predicted_file_path, 'w') as file:
    for response in predicted_responses:
        file.write(response + '\n')

with open(actual_file_path, 'w') as file:
    for response in actual_responses:
        file.write(response + '\n')

# Calculate the bit-wise accuracy
accuracy = (correct_bit_predictions / total_bits) * 100 if total_bits > 0 else 0
print(f"Total correct bit predictions: {correct_bit_predictions}, Total bits: {total_bits}")
print(f"Bit-wise accuracy: {accuracy:.2f}%")
print(f"Predicted responses saved to '{predicted_file_path}'.")
print(f"Actual responses saved to '{actual_file_path}'.")


# In[4]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

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
total_tests = len(X_test)

for idx in range(total_tests):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Check if the complete responses match, ignoring None predictions
    if np.array_equal(predicted_response[predicted_response != np.array(None)], 
                      actual_response[predicted_response != np.array(None)]):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_tests) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[1]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

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
total_tests = len(X_test)

for idx in range(total_tests):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Check if the complete responses match, ignoring None predictions
    if np.array_equal(predicted_response[predicted_response != np.array(None)], 
                      actual_response[predicted_response != np.array(None)]):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_tests) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[2]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

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
total_tests = len(X_test)

for idx in range(total_tests):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Check if the complete responses match, ignoring None predictions
    if np.array_equal(predicted_response[predicted_response != np.array(None)], 
                      actual_response[predicted_response != np.array(None)]):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_tests) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[3]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

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
total_tests = len(X_test)

for idx in range(total_tests):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Check if the complete responses match, ignoring None predictions
    if np.array_equal(predicted_response[predicted_response != np.array(None)], 
                      actual_response[predicted_response != np.array(None)]):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_tests) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[1]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

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
    # Check if the complete responses match, ignoring None predictions
    if np.array_equal(predicted_response[predicted_response != np.array(None)], 
                      actual_response[predicted_response != np.array(None)]):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_tests) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[5]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\first_10000_challenges.txt"
y_text_file = "D:\\data preperation\\New folder\\first_10000_responses.txt"
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
total_tests = len(X_test)

for idx in range(total_tests):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Check if the complete responses match, ignoring None predictions
    if np.array_equal(predicted_response[predicted_response != np.array(None)], 
                      actual_response[predicted_response != np.array(None)]):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_tests) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[9]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = []
    response_data = []

    for challenge in x_df['Challenge']:
        if isinstance(challenge, int):  # Convert integer to binary string if needed
            challenge = format(challenge, '032b')  # Assuming 32-bit binary numbers
        challenge_data.append([int(bit) for bit in challenge])

    for response in y_df['Y_Data']:
        if isinstance(response, int):
            response = format(response, '0128b')  # Assuming 128-bit binary numbers
        response_data.append([int(bit) for bit in response])

    return np.array(challenge_data), np.array(response_data)

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

# Load challenge and response data from files
x_text_file = "D:\\data preperation\\New folder\\first_10000_challenges1.txt"
y_text_file = "D:\\data preperation\\New folder\\first_10000_responses1.txt"
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
total_tests = len(X_test)

for idx in range(total_tests):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Check if the complete responses match, ignoring None predictions
    if np.array_equal(predicted_response[predicted_response != np.array(None)], 
                      actual_response[predicted_response != np.array(None)]):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_tests) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[1]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

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
total_tests = len(X_test)

for idx in range(total_tests):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Check if the complete responses match, ignoring None predictions
    if np.array_equal(predicted_response[predicted_response != np.array(None)], 
                      actual_response[predicted_response != np.array(None)]):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_tests) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[3]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(x_df, y_df):
    challenge_data = np.array([[int(bit) for bit in challenge] for challenge in x_df['Challenge']])
    response_data = np.array([[int(bit) for bit in response] for response in y_df['Y_Data']])
    return challenge_data, response_data

def train_models(challenge_data, response_data):
    models = []
    for i in range(response_data.shape[1]):
        if len(np.unique(response_data[:, i])) > 1:  # Ensure at least two classes
            model = SVC(kernel='linear')
            model.fit(challenge_data, response_data[:, i])
            models.append(model)
        else:
            models.append(None)  # No model for columns with insufficient class diversity
    return models

def predict_response(models, challenge):
    new_challenge_array = np.array([challenge])
    predicted_response = []
    for model in models:
        if model is not None:
            predicted_response.append(int(model.predict(new_challenge_array)[0]))
        else:
            predicted_response.append(None)  # Handle non-trained model slots
    return np.array(predicted_response)

# Load challenge and response data from files
x_text_file = "E:\\Study Oulu\\2. Second period\\Hardware Hacing\\Challenge.txt"
y_text_file = "E:\\Study Oulu\\2. Second period\\Hardware Hacing\\Response.txt"
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
total_tests = len(X_test)

for idx in range(total_tests):
    predicted_response = predict_response(models, X_test[idx])
    actual_response = y_test[idx]
    # Check if the complete responses match, ignoring None predictions
    if np.array_equal(predicted_response[predicted_response != np.array(None)], 
                      actual_response[predicted_response != np.array(None)]):
        correct_predictions += 1

# Calculate total accuracy
accuracy = (correct_predictions / total_tests) * 100
print(f"Total correct predictions: {correct_predictions}, Total tests: {total_tests}")
print(f"Overall accuracy: {accuracy:.2f}%")


# In[ ]:




