import cv2
import os


# Function to extract frames
def FrameCapture(path, output_dir):
    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1

    while success:
        # vidObj object calls read function extract frames
        success, image = vidObj.read()

        # Checks if the image has been retrieved successfully
        if not success:
            break

        # Saves the frames with frame-count as a six-digit number
        cv2.imwrite(f"{output_dir}/{count:06d}.png", image)

        count += 1


# Driver Code
if __name__ == '__main__':
    # The path to the video file
    video_path = r"C:\Users\cordo\Desktop\GitHub_2023_2024\Custom-Dataset-Complex-YOLOv4-Pytorch\src\my_custom_code\my_data\Zelzah and Plummer 1 2023-03-20 11-48-13.mkv"

    # The output directory
    output_dir = r"C:\Users\cordo\Desktop\GitHub_2023_2024\Custom-Dataset-Complex-YOLOv4-Pytorch\src\my_custom_code\my_data\frames_output"

    # Check if the directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calling the function
    FrameCapture(video_path, output_dir)
