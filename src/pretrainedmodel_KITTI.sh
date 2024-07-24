#!/bin/bash
(start_time=$(date +"%Y-%m-%d %H:%M:%S") && \
python3 train_custom_cpu.py --no_cuda --model_def config/cfg/complex_yolov4.cfg --pretrained_path ../checkpoints/complex_yolov4/complex_yolov4_mse_loss.pth --num_epochs 5 > output_$(date +"%Y-%m-%d").txt; \
end_time=$(date +"%Y-%m-%d %H:%M:%S") && \
echo "Start time: $start_time" && \
echo "End time: $end_time" && \
start_seconds=$(date -d "$start_time" +%s) && \
end_seconds=$(date -d "$end_time" +%s) && \
duration=$((end_seconds - start_seconds)) && \
echo "Duration: $duration seconds") 2>&1 | tee -a output_$(date +"%Y-%m-%d").txt
