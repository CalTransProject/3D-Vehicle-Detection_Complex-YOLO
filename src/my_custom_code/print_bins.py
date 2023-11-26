import struct
import os


def read_and_print_bin_file(binFileName):
    size_float = 4
    with open(binFileName, "rb") as f:
        while True:
            # Read 16 bytes (4 floats)
            bytes_read = f.read(size_float * 4)
            if not bytes_read:
                break  # Exit loop if end of file

            # Unpack the bytes to 4 floats
            x, y, z, intensity = struct.unpack("ffff", bytes_read)

            # Print the values
            print(f"x: {x}, y: {y}, z: {z}, intensity: {intensity}")


# Full path to the specific bin file
specific_bin_file = r"testing_purposes_data\bin\000000.bin"
# specific_bin_file = r"testing_purposes_data\bin_kitti\000000.bin"

# Check if the file exists before trying to read it
if os.path.exists(specific_bin_file):
    read_and_print_bin_file(specific_bin_file)
else:
    print(f"File not found: {specific_bin_file}")

