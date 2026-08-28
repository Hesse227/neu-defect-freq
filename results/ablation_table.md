# NEU-DET ablation results (test split, 180 images)

| model | params (M) | mAP@50 | mAP@50:95 | GPU inf (ms) | GPU FPS | CPU inf (ms) |
|---|---|---|---|---|---|---|
| YOLOv8n (baseline, seed 42) | 3.012 | 0.771 | 0.436 | 2.8 | 361.4 | 88.3 |
| YOLOv8n (baseline, seed 123) | 3.012 | 0.747 (-0.024) | 0.421 (-0.015) | 2.6 | 389.8 | 82.8 |
| YOLOv8n + DCT @ P3/P4/P5 (ours, seed 42) | 3.023 | 0.753 (-0.018) | 0.418 (-0.018) | 3.3 | 304.0 | 90.0 |
| YOLOv8n + DCT @ P3/P4/P5 (ours, seed 123) | 3.023 | 0.729 (-0.042) | 0.410 (-0.026) | 3.1 | 323.5 | 89.3 |
| YOLOv8n + DCT @ P3 only | 3.013 | 0.724 (-0.047) | 0.410 (-0.026) | 3.1 | 323.6 | 75.0 |
| YOLOv8n + SE @ P3/P4/P5 (control) | 3.023 | 0.723 (-0.048) | 0.401 (-0.035) | 3.1 | 323.6 | 92.8 |
| YOLOv8n + DCT @ P3/P4/P5, freeze first 5 layers | 3.023 | 0.732 (-0.039) | 0.412 (-0.024) | 3.0 | 336.3 | 86.8 |

## Per-class AP@50 (test split)

| model | crazing | inclusion | patches | pitted_surface | rolled-in_scale | scratches |
|---|---|---|---|---|---|---|
| YOLOv8n (baseline, seed 42) | 0.439 | 0.837 | 0.907 | 0.858 | 0.638 | 0.946 |
| YOLOv8n (baseline, seed 123) | 0.386 | 0.811 | 0.908 | 0.861 | 0.594 | 0.925 |
| YOLOv8n + DCT @ P3/P4/P5 (ours, seed 42) | 0.397 | 0.823 | 0.890 | 0.827 | 0.658 | 0.919 |
| YOLOv8n + DCT @ P3/P4/P5 (ours, seed 123) | 0.321 | 0.843 | 0.915 | 0.762 | 0.634 | 0.898 |
| YOLOv8n + DCT @ P3 only | 0.282 | 0.814 | 0.891 | 0.791 | 0.671 | 0.893 |
| YOLOv8n + SE @ P3/P4/P5 (control) | 0.296 | 0.837 | 0.903 | 0.755 | 0.647 | 0.902 |
| YOLOv8n + DCT @ P3/P4/P5, freeze first 5 layers | 0.294 | 0.808 | 0.913 | 0.782 | 0.670 | 0.922 |
