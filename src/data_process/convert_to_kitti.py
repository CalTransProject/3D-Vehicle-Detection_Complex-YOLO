import os
import numpy as np
import shutil
from tqdm import tqdm
import math
import glob

def create_kitti_directories(base_dir):
    """Create KITTI format directories"""
    dirs = [
        'training/image_2',
        'training/label_2',
        'training/velodyne',
        'training/calib',
        'ImageSets'
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(base_dir, 'kitti', dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")

def create_calib_file(calib_path):
    """Create a calibration file with default values"""
    calib_str = """P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.875744000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 4.485728000000e+01 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.163791000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.745884000000e-03
P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199936000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729905000000e-03
R0_rect: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01
Tr_velo_to_cam: 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01
Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01"""
    
    with open(calib_path, 'w') as f:
        f.write(calib_str)

def convert_label_to_kitti_format(label_line):
    """Convert a single label line from your format to KITTI format"""
    parts = label_line.strip().split()
    
    # Parse your format
    type_name = parts[0]  # Vehicle type
    truncation = float(parts[1])  # Using your truncation value
    occlusion = int(parts[2])  # Using your occlusion value
    alpha = float(parts[3])  # Using your alpha value
    
    # 3D box dimensions
    h = float(parts[7])  # height
    w = float(parts[8])  # width
    l = float(parts[9])  # length
    
    # 3D box location
    x = float(parts[10])  # x position
    y = float(parts[11])  # y position
    z = float(parts[12])  # z position
    
    # Rotation
    ry = float(parts[13])  # rotation_y
    
    # Calculate alpha if not directly available
    # alpha = ry + np.arctan2(z, x) - np.pi/2
    
    # For now, using dummy values for 2D bbox (you'll need to project 3D box to get real values)
    bbox_left = 0
    bbox_top = 0
    bbox_right = 100
    bbox_bottom = 100
    
    # Format string in KITTI format:
    # type truncation occlusion alpha bbox_left bbox_top bbox_right bbox_bottom height width length x y z rotation_y
    kitti_str = f"{type_name} {truncation:.2f} {occlusion} {alpha:.2f} {bbox_left:.2f} {bbox_top:.2f} "
    kitti_str += f"{bbox_right:.2f} {bbox_bottom:.2f} {h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f}"
    
    return kitti_str

def convert_dataset(source_base_dir, kitti_base_dir):
    """Convert the entire dataset to KITTI format"""
    # Create KITTI directories
    create_kitti_directories(kitti_base_dir)
    
    # Get paths
    image_source = os.path.join(source_base_dir, '2D Camera Images')
    lidar_source = os.path.join(source_base_dir, '3D Camera Binary Data')
    label_source = os.path.join(source_base_dir, 'Labels', 'Dataset Label Format from MATLAB (Yes Normalization of Rotation Angle) - Zelzah Avenue and Plummer Street 10_45 Mintutes')
    
    kitti_image_dir = os.path.join(kitti_base_dir, 'kitti', 'training', 'image_2')
    kitti_lidar_dir = os.path.join(kitti_base_dir, 'kitti', 'training', 'velodyne')
    kitti_label_dir = os.path.join(kitti_base_dir, 'kitti', 'training', 'label_2')
    kitti_calib_dir = os.path.join(kitti_base_dir, 'kitti', 'training', 'calib')
    
    # Get list of files
    label_files = sorted([f for f in os.listdir(label_source) if f.endswith('.txt')])
    
    for label_file in tqdm(label_files, desc="Converting dataset"):
        idx = int(label_file.split('.')[0])
        
        # Copy and convert label
        with open(os.path.join(label_source, label_file), 'r') as f:
            label_lines = f.readlines()
        
        kitti_labels = []
        for line in label_lines:
            kitti_label = convert_label_to_kitti_format(line)
            kitti_labels.append(kitti_label)
        
        with open(os.path.join(kitti_label_dir, f'{idx:06d}.txt'), 'w') as f:
            f.write('\n'.join(kitti_labels))
        
        # Copy image if it exists
        img_file = f'{idx:06d}.png'
        if os.path.exists(os.path.join(image_source, img_file)):
            shutil.copy2(
                os.path.join(image_source, img_file),
                os.path.join(kitti_image_dir, img_file)
            )
        
        # Copy velodyne data if it exists
        lidar_file = f'{idx:06d}.bin'
        if os.path.exists(os.path.join(lidar_source, lidar_file)):
            shutil.copy2(
                os.path.join(lidar_source, lidar_file),
                os.path.join(kitti_lidar_dir, lidar_file)
            )
        
        # Create calibration file
        calib_path = os.path.join(kitti_calib_dir, f"{idx:06d}.txt")
        create_calib_file(calib_path)
    
    # Create train/val split
    create_split_files(kitti_base_dir, label_files)

def create_split_files(kitti_base_dir, label_files):
    """Create train/val split files"""
    # Remove file extensions
    file_ids = sorted([f.split('.')[0] for f in label_files])
    
    # Shuffle the files
    np.random.shuffle(file_ids)
    
    # Split into train/val (80/20)
    split_idx = int(len(file_ids) * 0.8)
    train_ids = file_ids[:split_idx]
    val_ids = file_ids[split_idx:]
    
    # Write split files
    split_dir = os.path.join(kitti_base_dir, 'kitti', 'ImageSets')
    
    with open(os.path.join(split_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_ids))
    
    with open(os.path.join(split_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_ids))

def create_kitti_structure(base_dir):
    """Create KITTI dataset directory structure"""
    kitti_dir = os.path.join(base_dir, 'kitti_format')
    os.makedirs(os.path.join(kitti_dir, 'training', 'image_2'), exist_ok=True)
    os.makedirs(os.path.join(kitti_dir, 'training', 'velodyne'), exist_ok=True)
    os.makedirs(os.path.join(kitti_dir, 'training', 'label_2'), exist_ok=True)
    os.makedirs(os.path.join(kitti_dir, 'training', 'calib'), exist_ok=True)
    os.makedirs(os.path.join(kitti_dir, 'ImageSets'), exist_ok=True)
    return kitti_dir

def convert_dataset_zelzah(src_dir, kitti_dir):
    """Convert Zelzah dataset to KITTI format"""
    all_frames = []
    
    # Copy images
    img_src_dir = os.path.join(src_dir, '2D Camera Images')
    if os.path.exists(img_src_dir):
        for img in os.listdir(img_src_dir):
            if img.endswith('.jpg') or img.endswith('.png'):
                frame_id = os.path.splitext(img)[0]
                shutil.copy2(
                    os.path.join(img_src_dir, img),
                    os.path.join(kitti_dir, 'training', 'image_2', f'{frame_id}.png')
                )
                all_frames.append(frame_id)
    
    # Copy point clouds
    lidar_src_dir = os.path.join(src_dir, '3D Camera Binary Data')
    if os.path.exists(lidar_src_dir):
        for pc in os.listdir(lidar_src_dir):
            if pc.endswith('.bin'):
                frame_id = os.path.splitext(pc)[0]
                shutil.copy2(
                    os.path.join(lidar_src_dir, pc),
                    os.path.join(kitti_dir, 'training', 'velodyne', f'{frame_id}.bin')
                )
    
    # Convert and copy labels
    label_src_dir = os.path.join(src_dir, 'Labels')
    if os.path.exists(label_src_dir):
        # Find the actual label directory (it might be nested)
        for root, dirs, files in os.walk(label_src_dir):
            for file in files:
                if file.endswith('.txt'):
                    frame_id = os.path.splitext(file)[0]
                    convert_label_zelzah(
                        os.path.join(root, file),
                        os.path.join(kitti_dir, 'training', 'label_2', f'{frame_id}.txt')
                    )
    
    # Create calibration files
    for frame_id in all_frames:
        calib_path = os.path.join(kitti_dir, 'training', 'calib', f"{frame_id}.txt")
        create_calib_file(calib_path)
    
    # Create train/val split
    all_frames = sorted(list(set(all_frames)))
    np.random.shuffle(all_frames)
    split_idx = int(len(all_frames) * 0.8)
    
    with open(os.path.join(kitti_dir, 'ImageSets', 'train.txt'), 'w') as f:
        f.write('\n'.join(all_frames[:split_idx]))
    
    with open(os.path.join(kitti_dir, 'ImageSets', 'val.txt'), 'w') as f:
        f.write('\n'.join(all_frames[split_idx:]))

def convert_label_zelzah(src_path, dst_path):
    """Convert label from Zelzah format to KITTI format"""
    with open(src_path, 'r') as f:
        lines = f.readlines()
    
    kitti_labels = []
    for line in lines:
        data = line.strip().split()
        if len(data) < 15:  # Skip invalid lines
            continue
        
        # Get vehicle type from the data (assuming it's in the first position)
        # If not specified, default to 'Car'
        vehicle_type = data[0] if len(data) > 15 else 'Car'
        
        # Validate vehicle type is in our class list
        if vehicle_type not in ['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Motorcycle', 'SUV', 'Semi', 'Bus', 'Van']:
            vehicle_type = 'Car'  # Default to Car if unknown type
        
        # Convert to KITTI format:
        # type truncated occluded alpha bbox(4) dimensions(3) location(3) rotation_y
        kitti_line = [
            vehicle_type,  # type
            '0.00',  # truncated
            '0',    # occluded
            '0.00', # alpha
            '0.00', '0.00', '50.00', '50.00',  # bbox (placeholder)
            data[8], data[9], data[10],  # dimensions (h,w,l)
            data[11], data[12], data[13],  # location (x,y,z)
            data[14]  # rotation_y
        ]
        kitti_labels.append(' '.join(kitti_line))
    
    with open(dst_path, 'w') as f:
        f.write('\n'.join(kitti_labels))

if __name__ == '__main__':
    # Set your paths here
    base_dir = '/Users/jim2/Visual Studio Code/3D-Vehicle-Detection_Complex-YOLO/dataset/Zelzah Plummer'
    kitti_dir = create_kitti_structure(base_dir)
    
    # Get all source directories
    source_dirs = [d for d in os.listdir(base_dir) if d.startswith('Zelzah and Plummer')]
    
    # Convert each source directory
    for source_dir in source_dirs:
        print(f"Converting {source_dir}...")
        source_path = os.path.join(base_dir, source_dir)
        convert_dataset_zelzah(source_path, kitti_dir)
    
    print(f"Dataset converted to KITTI format at: {kitti_dir}")
