import numpy as np
import struct
import os


def read_pcd_file(pcdFileName):
    with open(pcdFileName, 'r') as f:
        # Skip the header lines
        while True:
            line = f.readline()
            if line.startswith('DATA'):
                break

        # Read the point data
        points = []
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:  # x, y, z, intensity
                try:
                    x, y, z, intensity = [float(p) for p in parts]
                    # Check for NaN values
                    if not any(np.isnan([x, y, z, intensity])):
                        points.append([x, y, z, intensity])
                except ValueError:
                    # Skip lines that can't be converted to floats (e.g., contains NaN)
                    continue

    return np.array(points)


def pcd_to_bin(points):
    list_bin = []
    for point in points:
        x, y, z, intensity = point
        # Normalize intensity from 0-255 to 0-1
        intensity_normalized = intensity / 255.0
        # Pack point data into binary format
        list_bin += struct.pack("ffff", x, y, z, intensity_normalized)

    return list_bin


def main(pcd_folder, bin_folder):
    print("PCD Folder:", pcd_folder)
    print("BIN Folder:", bin_folder)

    file_count = 0
    for filename in os.listdir(pcd_folder):
        if filename.endswith(".pcd"):
            pcd_file = os.path.join(pcd_folder, filename)
            bin_file = os.path.join(bin_folder, filename.replace(".pcd", ".bin"))

            points = read_pcd_file(pcd_file)
            binary_data = pcd_to_bin(points)

            # Write to .bin file
            with open(bin_file, "wb") as f:
                f.write(bytearray(binary_data))

            file_count += 1
            # if file_count >= 10:
            #     break
    print(f"Total files converted: {file_count}")


if __name__ == "__main__":
    pcd_folder = r"my_data_3d\pcd_data"
    bin_folder = r"my_data_3d\bin_data"
    main(pcd_folder, bin_folder)
