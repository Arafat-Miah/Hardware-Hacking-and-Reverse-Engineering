#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv

def calculate_xor(base, value):
    xor_result = base ^ value
    return xor_result

def check_duplicates(csv_file):
    data = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(int(row[0], 2))  # Convert binary string to integer
    num_values = len(data)

    with open('output.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        for i in range(num_values):
            base = data[i]
            xor_results = []
            for j in range(num_values):
                if j != i:
                    xor_result = calculate_xor(base, data[j])
                    xor_results.append(xor_result)
            if all(result == 0 for result in xor_results):
                writer.writerow([bin(base)[2:].zfill(128), 0])
            else:
                writer.writerow([bin(base)[2:].zfill(128), 1])

    print("Output file generated successfully!")

# Example usage
csv_file_path = input("Enter the path of the CSV file: ")
check_duplicates(csv_file_path)


# In[1]:


import csv

def calculate_xor(base, value):
    xor_result = base ^ value
    return xor_result

def check_duplicates(csv_file):
    data = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            binary_str = row[0]
            binary_value = int(binary_str, 2)
            data.append(binary_value)
    num_values = len(data)

    with open('output.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        for i in range(num_values):
            base = data[i]
            xor_results = []
            for j in range(num_values):
                if j != i:
                    xor_result = calculate_xor(base, data[j])
                    xor_results.append(xor_result)
            if all(result == 0 for result in xor_results):
                writer.writerow([format(base, '0128b'), 0])
            else:
                writer.writerow([format(base, '0128b'), 1])

    print("Output file generated successfully!")

# Example usage
csv_file_path = input("Enter the path of the CSV file: ")
check_duplicates(csv_file_path)


# In[3]:


import csv

# Function to calculate XOR of two bit strings
def calculate_xor(base, value):
    result = ''
    for i in range(len(base)):
        if base[i] == value[i]:
            result += '0'
        else:
            result += '1'
    return result

# Read CSV file and convert values to binary
data = []
with open('C:\\Users\\Asus\\Desktop\Book3.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        value = bin(int(row[0]))[2:].zfill(128)  # Convert decimal to binary
        data.append(value)

# Compare values and calculate XOR
output = []
for i in range(len(data)):
    base = data[i]
    xor_result = '0' * 128  # Initialize XOR result to all zeros
    for j in range(len(data)):
        if j != i:
            xor_result = calculate_xor(base, data[j])
            if xor_result == '0' * 128:
                break
    output.append(xor_result)

# Write output to a new file
with open('output.csv', 'w') as file:
    writer = csv.writer(file)
    for row in output:
        writer.writerow([row])


# In[4]:


import csv

# Function to calculate XOR of two bit strings
def calculate_xor(base, value):
    result = ''
    for i in range(len(base)):
        if base[i] == value[i]:
            result += '0'
        else:
            result += '1'
    return result

# Read CSV file and convert values to binary
data = []
with open('C:\\Users\\Asus\\Desktop\Book3.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    for row in reader:
        value = bin(int(row[0]))[2:].zfill(128)  # Convert decimal to binary
        data.append(value)

# Compare values and calculate XOR
output = []
for i in range(len(data)):
    base = data[i]
    xor_result = '0' * 128  # Initialize XOR result to all zeros
    for j in range(len(data)):
        if j != i:
            xor_result = calculate_xor(base, data[j])
            if xor_result == '0' * 128:
                break
    output.append(xor_result)

# Write output to a new file
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in output:
        writer.writerow([row])


# In[6]:


import csv

# Function to calculate XOR of two bit strings
def calculate_xor(base, value):
    result = ''
    for i in range(len(base)):
        if base[i] == value[i]:
            result += '0'
        else:
            result += '1'
    return result

# Read CSV file and convert values to binary
data = []
with open('C:\\Users\\Asus\\Desktop\Book3.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        value = row[0].zfill(128)  # Pad with leading zeros if necessary
        data.append(value)

# Compare values and calculate XOR
output = []
for i in range(len(data)):
    base = data[i]
    xor_result = '0' * 128  # Initialize XOR result to all zeros
    for j in range(len(data)):
        if j != i:
            xor_result = calculate_xor(base, data[j])
            if xor_result == '0' * 128:
                break
    output.append(xor_result)

# Write output to a new file
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in output:
        writer.writerow([row])


# In[7]:


import os
print(os.getcwd())  # Print the current working directory


# In[1]:


import pandas as pd

# read the text file (one binary string per line)
txt_file = r'E:\\Study Oulu\\2. Second period\\Hardware Hacing\\bits.txt'
csv_file = r'E:\\Study Oulu\\2. Second period\\Hardware Hacing\\bits.csv'

# load text file
data = pd.read_csv(txt_file, header=None)

# save as csv
data.to_csv(csv_file, index=False, header=False)

print("CSV file created successfully!")


# In[2]:


import csv

# Function to calculate XOR of two bit strings
def calculate_xor(base, value):
    result = ''
    for i in range(len(base)):
        if base[i] == value[i]:
            result += '0'
        else:
            result += '1'
    return result

# Read CSV file and convert values to binary
data = []
with open('E:\\Study Oulu\\2. Second period\\Hardware Hacing\\bits.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        value = row[0].zfill(128)  # Pad with leading zeros if necessary
        data.append(value)

# Compare values and calculate XOR
output = []
for i in range(len(data)):
    base = data[i]
    xor_result = '0' * 128  # Initialize XOR result to all zeros
    for j in range(len(data)):
        if j != i:
            xor_result = calculate_xor(base, data[j])
            if xor_result == '0' * 128:
                break
    output.append(xor_result)

# Write output to a new file
with open('E:\\Study Oulu\\2. Second period\\Hardware Hacing\\output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in output:
        writer.writerow([row])


# In[3]:


import csv

def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

# read data
data = []
with open(r'E:\\Study Oulu\\2. Second period\\Hardware Hacing\\bits.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row[0])

N = 128
M = len(data)

total_hd = 0
count = 0

for i in range(M):
    for j in range(i+1, M):
        hd = hamming_distance(data[i], data[j])
        total_hd += hd
        count += 1

uniqueness = (2 * total_hd) / (M * (M - 1) * N) * 100

print(f"Uniqueness = {uniqueness:.2f}%")


# In[ ]:




