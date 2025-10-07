import os

# Define the file path
file_path = '/home/results/checkpoints/test.txt'

# Extract the directory path from the file path
directory = os.path.dirname(file_path)

# Check if the directory exists, and create it if it doesn't
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created.")
else:
    print(f"Directory '{directory}' already exists.")

# Write a test text file
with open(file_path, 'w') as file:
    file.write("This is a test text file.")

print(f"Test text file written to '{file_path}'.")

# Read the content of the text file
with open(file_path, 'r') as file:
    content = file.read()

# Write the content to the console
print("Content of the text file:")
print(content)
