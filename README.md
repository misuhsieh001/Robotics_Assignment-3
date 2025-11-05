# Robotics_Assignment-3

# Quick start

```bash
cd ~
git clone https://github.com/misuhsieh001/Robotics_Assignment-3.git
```


For Part A, just execute hw_a.py using
```bash
cd ~/HW3
python3 ./hw3_a.py
```

For Part B, execute
```bash
cd ~/HW3
python3 hw3_b.py cubes.png
```


For Part C, execute
```bash
python3 hw3_c.py cubes.png depth.png
```



```python
# Two-stage morphology (erode -> dilation then dilation -> erode)
    kernel_open = np.ones((3, 3), np.uint8) # Setup kernel for morphology operations, the more the stronger.
    mask_opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open) # Remove noise: erode -> dilation
    kernel_close = np.ones((7, 7), np.uint8) # Larger kernel to fill holes, the more the stronger.
    cleaned_mask = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close) # Fill holes: dilation -> erode
```

