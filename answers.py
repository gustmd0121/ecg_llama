import csv
from collections import Counter

def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return [row['class'].lower() for row in reader]

def write_to_file(file_name, data, count):
    with open(file_name, 'w') as f:
        f.write(f"Total count: {count}\n\n")
        for item in sorted(data):
            f.write(f"{item}\n")

# Read the CSV files
ptbxl_answers = set(read_csv('/home/hschung/llama-multimodal-vqa/ptbxl_answers.csv'))
mimic_answers = set(read_csv('/home/hschung/llama-multimodal-vqa/mimic_answers.csv'))

# Find unique and common answers
ptbxl_unique = ptbxl_answers - mimic_answers
mimic_unique = mimic_answers - ptbxl_answers
common_answers = ptbxl_answers.intersection(mimic_answers)

# Write results to files
write_to_file('ptbxl_unique.txt', ptbxl_unique, len(ptbxl_unique))
write_to_file('mimic_unique.txt', mimic_unique, len(mimic_unique))
write_to_file('common_answers.txt', common_answers, len(common_answers))

# Print summary
print(f"Unique to PTB-XL: {len(ptbxl_unique)}")
print(f"Unique to MIMIC: {len(mimic_unique)}")
print(f"Common to both: {len(common_answers)}")
print(f"Total in PTB-XL: {len(ptbxl_answers)}")
print(f"Total in MIMIC: {len(mimic_answers)}")