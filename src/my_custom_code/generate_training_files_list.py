import os

# Specify the path to the training directory
training_dir = '../../dataset/custom/training/label_2'

# Specify the filename for the output .txt file
output_file_name = 'train_revised.txt'

# Specify the path for the output .txt file that will list the names
output_dir = '../../dataset/custom/ImageSets'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Full path to the output file
output_file_path = os.path.join(output_dir, output_file_name)

# Get a list of all .txt files in the training directory
txt_files = [f for f in os.listdir(training_dir) if f.endswith('.txt')]

# Extract base names without the extension and sort them
base_names = sorted([os.path.splitext(f)[0] for f in txt_files])

# Write the base names to the output file
with open(output_file_path, 'w') as file_list:
    for name in base_names:
        file_list.write(name + '\n')

print(f'File list has been written to {output_file_path}')
