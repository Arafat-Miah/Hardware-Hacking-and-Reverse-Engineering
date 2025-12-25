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




