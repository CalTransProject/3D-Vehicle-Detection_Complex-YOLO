#!/bin/bash

# Record the start time
start_time=$(date +"%Y-%m-%d %H:%M:%S") && \

# Execute the Python script using time command to track its execution time and log it
python3 train_custom_cpu.py --no_cuda --model_def config/cfg/complex_yolov4.cfg --pretrained_path ../checkpoints/complex_yolov4/complex_yolov4_mse_loss.pth --num_epochs 5 > output_$(date +"%Y-%m-%d").txt; \

# Record the end time
end_time=$(date +"%Y-%m-%d %H:%M:%S") && \

# Display start and end times
echo "Start time: $start_time" && \
echo "End time: $end_time" && \

# Calculate the duration of the script execution in seconds
start_seconds=$(date -d "$start_time" +%s) && \
end_seconds=$(date -d "$end_time" +%s) && \
duration=$((end_seconds - start_seconds)) && \
echo "Duration: $duration seconds" 2>&1 | tee -a output_$(date +"%Y-%m-%d").txt