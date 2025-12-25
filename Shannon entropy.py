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




