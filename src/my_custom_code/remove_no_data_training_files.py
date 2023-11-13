import os

# Specify the paths to the directories
txt_dir = '../../dataset/custom/training/label_2'
additional_txt_dir = '../../dataset/custom/training/calib'
png_dir = '../../dataset/custom/training/image_2'
bin_dir = '../../dataset/custom/training/velodyne'

# Print the paths
print(f"Text directory: {txt_dir}")
print(f"Additional text directory: {additional_txt_dir}")
print(f"PNG directory: {png_dir}")
print(f"BIN directory: {bin_dir}")

# Gather all the .txt files in the txt_dir
txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

# Loop over each file
for file in txt_files:
    file_path = os.path.join(txt_dir, file)

    # Check if the .txt file is empty
    if os.path.getsize(file_path) == 0:
        # Get the base file name without the extension
        base_name = os.path.splitext(file)[0]

        # Construct the file paths for the corresponding .png, .bin files, and the additional .txt file
        png_file_path = os.path.join(png_dir, base_name + '.png')
        bin_file_path = os.path.join(bin_dir, base_name + '.bin')
        additional_txt_file_path = os.path.join(additional_txt_dir, file)

        # Remove the .txt file from the original directory
        print(f"Deleting empty .txt file: {file}")
        os.remove(file_path)

        # Remove the corresponding .png file if it exists
        if os.path.exists(png_file_path):
            print(f"Deleting associated .png file: {os.path.basename(png_file_path)}")
            os.remove(png_file_path)

        # Remove the corresponding .bin file if it exists
        if os.path.exists(bin_file_path):
            print(f"Deleting associated .bin file: {os.path.basename(bin_file_path)}")
            os.remove(bin_file_path)

        # Remove the corresponding .txt file from the additional directory if it exists
        if os.path.exists(additional_txt_file_path):
            print(f"Deleting additional .txt file: {os.path.basename(additional_txt_file_path)}")
            os.remove(additional_txt_file_path)
