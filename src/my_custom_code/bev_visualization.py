import numpy as np
import cv2
import os


import numpy as np
import math

class Object3D(object):
    '''3d object label'''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        self.type = data[0]  # Object type, e.g., 'Car', 'Motorcycle', ...
        self.truncation = float(data[1])  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        # Removed direct assignment to self.alpha from data[3]
        self.xctr = float(data[3])  # Object's center x
        self.yctr = float(data[4])  # Object's center y
        self.zctr = float(data[5])  # Object's center z
        self.xlen = float(data[6])  # Object's length in x
        self.ylen = float(data[7])  # Object's height in y
        self.zlen = float(data[8])  # Object's width in z
        self.xrot = float(data[9])  # Rotation around x-axis
        self.yrot = float(data[10])  # Rotation around y-axis (ry)
        self.zrot = float(data[11])  # Rotation around z-axis

        self.h = self.ylen  # Height of the bounding box
        self.w = self.zlen  # Width of the bounding box
        self.l = self.xlen  # Length of the bounding box

        # self.ry = -self.zrot - np.pi / 2
        # self.ry = self.zrot  # Yaw rotation
        # Normalize zrot to be within [-pi, pi] for yaw rotation
        # self.ry = ((self.zrot + np.pi) % (2 * np.pi)) - np.pi

        r_y = -(self.zrot) - np.pi / 2  # Your conversion here
        r_y_normalized = self.normalize_angle(r_y)
        self.ry = r_y_normalized

        # Compute alpha based on the provided rotation_y and object's X position
        self.alpha = self.compute_alpha(self.ry, self.xctr)

        self.cls_id = self.cls_type_to_id(self.type)
        self.dis_to_cam = np.linalg.norm(np.array([self.xctr, self.yctr, self.zctr]))
        self.score = -1.0  # Placeholder for detection score
        self.level_str = 'Easy'  # Placeholder for difficulty level
        self.level = 1  # Placeholder for level ID

    # def compute_alpha(self, rotation_y, x):
    #     '''Compute alpha angle based on rotation_y and object's X position.'''
    #     theta = math.atan2(x, self.zctr)
    #     alpha = rotation_y - theta
    #     # Normalize alpha to be within [-pi, pi]
    #     alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
    #     return alpha


    def compute_alpha(self, rotation_y_degrees, x):
        """Compute alpha angle based on rotation_y in degrees and object's X position."""
        # Convert rotation_y from degrees to radians
        rotation_y_radians = math.radians(rotation_y_degrees)

        # Calculate theta in radians
        theta = math.atan2(x, self.zctr)

        # Calculate alpha in radians
        alpha_radians = rotation_y_radians - theta

        # Normalize alpha to be within [-π, π]
        alpha_radians = (alpha_radians + np.pi) % (2 * np.pi) - np.pi

        # Optional: Convert alpha back to degrees for other calculations or output
        alpha_degrees = math.degrees(alpha_radians)

        # Return both radians and degrees if needed, or just one of them
        # return alpha_radians, alpha_degrees
        return alpha_radians  # If only radians are needed

    def cls_type_to_id(self, cls_type):
        '''Map class type to an ID.'''
        CLASS_NAME_TO_ID = {
            'Car': 0,
            'Pedestrian': 1,
            'Cyclist': 2,
            # Additional mappings can be added here
        }
        return CLASS_NAME_TO_ID.get(cls_type, -1)

    def normalize_angle(self, angle):
        """Normalize an angle to be within [-π, π]"""
        normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return normalized_angle





def load_labels(label_filename):
    objects = []
    with open(label_filename, 'r') as file:
        for line in file:
            obj = Object3D(line)
            objects.append(obj)
    return objects


def draw_3d_box(bev_image, obj, boundary, discretization=(0.1, 0.1)):
    # Skip drawing if the object is out of the specified boundary
    if not (boundary['minX'] <= obj.xctr <= boundary['maxX'] and boundary['minY'] <= obj.yctr <= boundary['maxY']):
        return

    # Convert the object's center position to BEV pixel coordinates
    x = int((obj.xctr - boundary['minX']) / discretization[0])
    y = int((obj.yctr - boundary['minY']) / discretization[1])

    # Convert object dimensions to BEV scale
    length = int(obj.l / discretization[0])
    width = int(obj.w / discretization[1])

    # Draw the bounding box on the BEV image
    cv2.rectangle(bev_image, (x - length // 2, y - width // 2),
                  (x + length // 2, y + width // 2), (0, 255, 0), 2)


def load_velo_scan(velo_filename):
    """
    Load and parse a LiDAR scan from a binary file.
    """
    scan = np.fromfile(velo_filename, dtype=np.float32)
    return scan.reshape((-1, 4))


def create_bev_image(lidar_data, boundary, discretization=(0.1, 0.1)):
    """
    Converts LiDAR data to a BEV image within specified boundaries.
    """
    width = int((boundary['maxX'] - boundary['minX']) / discretization[0])
    height = int((boundary['maxY'] - boundary['minY']) / discretization[1])
    bev_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Filter points within boundaries
    indices = np.where(
        (lidar_data[:, 0] >= boundary['minX']) & (lidar_data[:, 0] <= boundary['maxX']) &
        (lidar_data[:, 1] >= boundary['minY']) & (lidar_data[:, 1] <= boundary['maxY'])
    )[0]

    # Transform LiDAR coordinates to BEV map coordinates
    x_coords = np.int_((lidar_data[indices, 0] - boundary['minX']) / discretization[0])
    y_coords = np.int_((lidar_data[indices, 1] - boundary['minY']) / discretization[1])

    # Visualization - the brighter the point, the higher it is
    for x, y, z in zip(x_coords, y_coords, lidar_data[indices, 2]):
        color = min(int((z - boundary['minZ']) / (boundary['maxZ'] - boundary['minZ']) * 255), 255)
        bev_image[y, x] = (color, color, color)

    return bev_image


def add_labels_to_bev(bev_image, labels, boundary, discretization=(0.1, 0.1)):
    """
    Adds labels to the BEV image using rotated boxes to account for object orientation.
    """
    for label in labels:
        # Extract label coordinates (assuming labels are in the format [x, y, z, l, w, h, yaw])
        x, y, z, l, w, h, yaw = label

        # Check if the label is within the boundary
        if boundary['minX'] <= x <= boundary['maxX'] and boundary['minY'] <= y <= boundary['maxY']:
            # Convert to BEV coordinates
            bev_x = (x - boundary['minX']) / discretization[0]
            bev_y = (y - boundary['minY']) / discretization[1]

            # Convert dimensions to BEV scale
            bev_l = l / discretization[0]
            bev_w = w / discretization[1]

            # Convert yaw angle from degrees to radians if necessary
            yaw_rad = np.deg2rad(yaw)  # Remove this line if yaw is already in radians

            # Draw the rotated box on the BEV image
            drawRotatedBox(bev_image, bev_x, bev_y, bev_w, bev_l, yaw_rad, color=(0, 255, 0))

    return bev_image

def get_corners(x, y, width, length, yaw):
    """
    Calculate the corners of a given box in BEV space.

    Parameters:
    - x, y: Center coordinates of the box
    - width, length: Size of the box
    - yaw: Rotation angle of the box in radians

    Returns:
    - corners: Coordinates of the box corners
    """
    # Calculate rotation matrix
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])

    # Define corners in local box coordinates
    local_corners = np.array([
        [length / 2, width / 2], [length / 2, -width / 2],
        [-length / 2, -width / 2], [-length / 2, width / 2]
    ])

    # Rotate and translate corners
    corners = np.dot(local_corners, rotation_matrix.T) + np.array([x, y])

    return corners

def drawRotatedBox(img, x, y, width, length, yaw, color=(0, 255, 0), thickness=2):
    """
    Draw a rotated box on the BEV image.

    Parameters:
    - img: The BEV image
    - x, y: Center coordinates of the box in BEV space
    - width, length: Size of the box
    - yaw: Rotation angle of the box in radians
    - color: Box color
    - thickness: Line thickness
    """
    corners = get_corners(x, y, width, length, yaw)
    corners_int = np.int0(corners)  # Convert to integer for OpenCV functions

    # Draw lines between each corner to form the box
    for i in range(4):
        cv2.line(img, tuple(corners_int[i]), tuple(corners_int[(i + 1) % 4]), color, thickness)

    return img

def main():
    dataset_dir = "../../dataset/custom/training"
    sample_id = "000100"
    lidar_file = os.path.join(dataset_dir, "velodyne", f"{sample_id}.bin")
    label_file = os.path.join(dataset_dir, "label_2", f"{sample_id}.txt")

    # Load LiDAR data
    lidar_data = load_velo_scan(lidar_file)

    # Define the boundary for the BEV image (in meters)
    boundary = {'minX': -10, 'maxX': 30, 'minY': -10, 'maxY': 10, 'minZ': -2, 'maxZ': 2}

    # Create BEV image
    bev_image = create_bev_image(lidar_data, boundary)

    # Load labels from the label file
    objects = []  # This will hold the parsed Object3D instances
    with open(label_file, 'r') as file:
        for line in file:
            obj = Object3D(line)  # Assuming Object3D class is defined as you provided
            objects.append(obj)

    # Add labels to BEV image based on parsed Object3D instances
    # for obj in objects:
    #     # Convert 3D object properties to a format suitable for 2D BEV representation
    #     # For simplicity, using the center x, y and dimensions length, width directly
    #     # You might want to adjust based on the object's orientation (rotation_y, alpha)
    #     label = [obj.xctr, obj.yctr, obj.zctr, obj.l, obj.w, obj.h]
    #     bev_image_with_labels = add_labels_to_bev(bev_image, [label], boundary)

    # Prepare label data for all objects
    # labels = [[obj.xctr, obj.yctr, obj.zctr, obj.l, obj.w, obj.h] for obj in objects]
    # Prepare label data for all objects including yaw angle
    labels = [[obj.xctr, obj.yctr, obj.zctr, obj.l, obj.w, obj.h, obj.ry] for obj in objects]

    # Add all labels to BEV image at once with rotation considered
    bev_image_with_labels = add_labels_to_bev(bev_image, labels, boundary)

    # Save or display the BEV image with labels
    cv2.imwrite(f"bev_image_with_labels_{sample_id}.png", bev_image_with_labels)
    cv2.imshow("BEV Image with Labels", bev_image_with_labels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
