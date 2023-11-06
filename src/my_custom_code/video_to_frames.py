import cv2
import os


# Function to extract frames
def FrameCapture(path, output_dir, frames_per_second=10):
    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Getting the frames per second (fps) of the video
    fps = int(vidObj.get(cv2.CAP_PROP_FPS))

    # Calculating the frame skip interval
    frame_skip_interval = int(fps / frames_per_second)

    # Counter used for jumping over the Frames
    count = 0

    # Frame counter for image file name count
    img_count = 0

    # checks whether frames were extracted
    # success = 1
    success = True

    while success:
        # vidObj object calls read function extract frames
        success, image = vidObj.read()

        # Checks if the image has been retrieved successfully
        if not success:
            break

        # Saves the frames with frame-count as a six-digit number
        # cv2.imwrite(f"{output_dir}/{count:06d}.png", image)

        # Only save every 'frame_skip_interval'-th frame
        if success and count % frame_skip_interval == 0:
            # Frame is saved with a six-digit number
            cv2.imwrite(f"{output_dir}/{img_count:06d}.png", image)
            img_count += 1

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
