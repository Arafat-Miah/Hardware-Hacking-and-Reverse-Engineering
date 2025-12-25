#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os


# In[1]:


a=1
b=2
c=a+b
print(c)


# In[2]:


pwd


# In[1]:


import pandas as pd
df=pd.read_csv("arafat_faisal.csv")
df


# In[3]:


import csv
import pandas as pd
data = pd.read_csv('C:\\Users\\Asus\\Downloads\\EVs and PUF\\Data\\out crp\\Shakil vai\CRP1.csv')
data


# In[4]:


import csv
import pandas as pd

# Set display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('C:\\Users\\Asus\\Downloads\\EVs and PUF\\Data\\out crp\\Shakil vai\CRP1.csv')

# Print the DataFrame
import math
from collections import Counter

# Assuming your DataFrame variable is named 'data'

# Concatenate all values in the DataFrame into a single string
data_concatenated = ''.join(data.iloc[:, 0])

# Calculate the frequency of each unique character
character_counts = Counter(data_concatenated)

# Calculate the total number of characters
total_characters = len(data_concatenated)

# Calculate the probability of each character
character_probabilities = [count / total_characters for count in character_counts.values()]

# Calculate the Shannon entropy
entropy = -sum(p * math.log2(p) for p in character_probabilities)

# Print the entropy
print("Shannon Entropy:", entropy)



# In[5]:


import csv
import pandas as pd

# Set display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('C:\\Users\\Asus\\Downloads\\EVs and PUF\\Data\\out crp\\Shakil vai\CRP1.csv')

# Print the DataFrame
import math
from collections import Counter

# Assuming your DataFrame variable is named 'data'

# Concatenate all values in the DataFrame into a single string
data_concatenated = ''.join(data.iloc[:, 0])

# Calculate the frequency of each unique character
character_counts = Counter(data_concatenated)

# Calculate the total number of characters
total_characters = len(data_concatenated)

# Calculate the probability of each character
character_probabilities = [count / total_characters for count in character_counts.values()]

# Calculate the Shannon entropy
entropy = -sum(p * math.log2(p) for p in character_probabilities)

# Print the entropy
print("Shannon Entropy:", entropy)



# In[6]:


import math
import pandas as pd

# Assuming your DataFrame variable is named 'data'

# Read the DataFrame
data = pd.read_csv('C:\\Users\\Asus\\Downloads\\EVs and PUF\\Data\\out crp\\Shakil vai\CRP1.csv')

# Extract the binary values from the DataFrame
binary_values = data.iloc[:, 0]

# Calculate and store the entropy values for each binary value
entropy_values = []
for binary_value in binary_values:
    character_counts = {}
    total_characters = 0
    
    # Calculate the frequency of each unique character in the binary value
    for bit in binary_value:
        if bit in character_counts:
            character_counts[bit] += 1
        else:
            character_counts[bit] = 1
        total_characters += 1
    
    # Calculate the probability of each character
    character_probabilities = [count / total_characters for count in character_counts.values()]
    
    # Calculate the Shannon entropy
    entropy = -sum(p * math.log2(p) for p in character_probabilities)
    
    # Store the entropy value
    entropy_values.append(entropy)

# Save the entropy values to a file
output_file = 'entropy_values.csv'
pd.DataFrame({'Entropy': entropy_values}).to_csv(output_file, index=False)

# Calculate and display the overall Shannon entropy
overall_entropy = sum(entropy_values) / len(entropy_values)
print("Overall Shannon Entropy:", overall_entropy)


# In[7]:


import os

# Get the current working directory
current_directory = os.getcwd()

# Print the current working directory
print("Current directory:", current_directory)


# In[2]:


import math
import pandas as pd

# Assuming your DataFrame variable is named 'data'

# Read the DataFrame
data = pd.read_excel('C:\\Users\\Asus\\Desktop\Book4.xlsx')

# Extract the binary values from the DataFrame
binary_values = data.iloc[:, 0]

# Calculate and store the entropy values for each binary value
entropy_values = []
for binary_value in binary_values:
    character_counts = {}
    total_characters = 0
    
    # Calculate the frequency of each unique character in the binary value
    for bit in binary_value:
        if bit in character_counts:
            character_counts[bit] += 1
        else:
            character_counts[bit] = 1
        total_characters += 1
    
    # Calculate the probability of each character
    character_probabilities = [count / total_characters for count in character_counts.values()]
    
    # Calculate the Shannon entropy
    entropy = -sum(p * math.log2(p) for p in character_probabilities)
    
    # Store the entropy value
    entropy_values.append(entropy)

# Save the entropy values to a file
output_file = 'entropy_values1.csv'
pd.DataFrame({'Entropy': entropy_values}).to_csv(output_file, index=False)

# Calculate and display the overall Shannon entropy
overall_entropy = sum(entropy_values) / len(entropy_values)
print("Overall Shannon Entropy:", overall_entropy)


# In[4]:


import math
import pandas as pd

# Assuming your DataFrame variable is named 'data'

# Read the DataFrame
data = pd.read_excel('E:\\Study Oulu\\2. Second period\\Hardware Hacing\\bits.xlsx')

# Extract the binary values from the DataFrame
binary_values = data.iloc[:, 0]

# Calculate and store the entropy values for each binary value
entropy_values = []
for binary_value in binary_values:
    character_counts = {}
    total_characters = 0
    
    # Calculate the frequency of each unique character in the binary value
    for bit in binary_value:
        if bit in character_counts:
            character_counts[bit] += 1
        else:
            character_counts[bit] = 1
        total_characters += 1
    
    # Calculate the probability of each character
    character_probabilities = [count / total_characters for count in character_counts.values()]
    
    # Calculate the Shannon entropy
    entropy = -sum(p * math.log2(p) for p in character_probabilities)
    
    # Store the entropy value
    entropy_values.append(entropy)

# Save the entropy values to a file
output_file = 'E:\\Study Oulu\\2. Second period\\Hardware Hacing\entropy_values.csv'
pd.DataFrame({'Entropy': entropy_values}).to_csv(output_file, index=False)

# Calculate and display the overall Shannon entropy
overall_entropy = sum(entropy_values) / len(entropy_values)
print("Overall Shannon Entropy:", overall_entropy)


# In[ ]:




