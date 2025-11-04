import cv2
import sys
import numpy as np
import math

def get_valid_depth(depth_img, u, v, w=10): # Use a small window to find valid depth, because depth images can be noisy according to the PDF.

    # Get integer coordinates for slicing
    u_int, v_int = int(u), int(v)
    
    # Define a small window (e.g., 5x5) around the centroid
    half_w = w // 2
    
    # Get the sub-window from the depth image
    window = depth_img[v_int - half_w : v_int + half_w + 1, 
                       u_int - half_w : u_int + half_w + 1]
    # print(window)
    # Find all non-zero (valid) depth values in the window 
    valid_depths = window[window > 0]
    if valid_depths.size > 0:
        # If we have valid depths, return the average
        return np.mean(valid_depths)
    else:
        # If the entire window is 0 (invalid), return 0
        return 0.0

def find_objects_and_properties(img, depth_img, K):
    
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Create color ranges ---
    # These are the HSV ranges I found worked in the last step.
    # You can also tweak them further if needed.
    # But do it carefully to avoid losing accuracy !!!!
    color_ranges = {
        'red1': (np.array([0, 130, 60]), np.array([10, 200, 150])),
        'red2': (np.array([170, 50, 70]), np.array([180, 100, 200])),
        'green': (np.array([50, 50, 50]), np.array([70, 180, 255])),
        'blue': (np.array([90, 60, 30]), np.array([130, 255, 200])),
        'yellow': (np.array([10, 80, 100]), np.array([40, 200, 255])),
        'white_gray': (np.array([0, 0, 60]), np.array([180, 90, 255])),
    }
    
    # For red channel, we need to combine two ranges.
    combined_mask = np.zeros(img.shape[:2], dtype="uint8")

    # --- Create masks ---
    # Iterate through each color range and create masks.
    # Reference: https://steam.oxxostudio.tw/category/python/ai/opencv-inrange.html
    # Reference: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
    for color_name, (lower, upper) in color_ranges.items(): # Iterate through each color range
        mask = cv2.inRange(hsv_image, lower, upper)
        if color_name == 'red1':
            mask_red2 = cv2.inRange(hsv_image, color_ranges['red2'][0], color_ranges['red2'][1])
            # Use bitwise OR to combine both red masks, you must use bitwise_or here.
            mask = cv2.bitwise_or(mask, mask_red2)
        elif color_name == 'red2':
            continue
        # Use bitwise OR to combine all color masks, you must use bitwise_or here too.
        combined_mask = cv2.bitwise_or(combined_mask, mask) #

    # Two-stage morphology (erode -> dilation then dilation -> erode)
    kernel_open = np.ones((3, 3), np.uint8) # Setup kernel for morphology operations, the more the stronger.
    mask_opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open) # Remove noise: erode -> dilation
    kernel_close = np.ones((7, 7), np.uint8) # Larger kernel to fill holes, the more the stronger.
    cleaned_mask = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close) # Fill holes: dilation -> erode
    
    # Find contours (keep CHAIN_APPROX_NONE)
    # Use RETR_EXTERNAL to get only outer contours.
    # Use CHAIN_APPROX_NONE to keep all contour points. (high accuracy but memory consuming)
    # Reference: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
    # Reference: https://chtseng.wordpress.com/2016/12/05/opencv-contour%E8%BC%AA%E5%BB%93/
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE) 

    detected_results = [] 
    
    # --- NEW: Get K-Matrix parameters ---
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Process each contour
    for i, cnt in enumerate(contours):
        # Filter small contours, it's important to avoid noise.
        if cv2.contourArea(cnt) < 500: 
            continue

        # Reference for cv2.moments: https://chtseng.wordpress.com/2016/12/05/opencv-contour%E8%BC%AA%E5%BB%93/
        # There is a guildline to get centroid from moments.
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
            
        # 2D Centroid (u, v)
        u = M["m10"] / M["m00"]
        v = M["m01"] / M["m00"]
        print(f"Detected object at 2D centroid: ({u:.2f}, {v:.2f})")
        # Draw a small circle at the 2D centroid on the image for visualization
        u_int, v_int = int(round(u)), int(round(v))
        h_img, w_img = img.shape[:2]
        if 0 <= u_int < w_img and 0 <= v_int < h_img:
            cv2.circle(img, (u_int, v_int), 2, (1, 227, 254), -1)
        
        # --- Part C: 3D Calculation ---
        
        # 1. Get Z_c (Depth) from the depth image 
        Z_c = get_valid_depth(depth_img, u, v)
        
        if Z_c == 0.0:
            # Cannot calculate 3D position if depth is invalid
            continue
            
        # 2. Calculate X_c and Y_c using inverse projection
        X_c = (u - cx) * Z_c / fx
        Y_c = (v - cy) * Z_c / fy
        
        # Store 3D results
        detected_results.append({
            'id': -1, 
            'centroid_2d': (u, v),
            'coord_3d': (X_c, Y_c, Z_c)
        })

    # --- Part B: ID Matching (Same as Fixed 14) ---
    
    # PDF ID and their approximate 2D centroids
    pdf_ids = {
        0: (170, 216), # Red
        1: (364, 208), # White
        2: (387, 91),  # Blue
        3: (163, 94),  # Green
        4: (245, 57)   # Yellow
    }
    # Create a list to hold final matched objects.
    final_objects = [None] * 5
    # This part of the code is to match detected objects to PDF IDs based on 2D centroid proximity.
    for obj in detected_results:
        min_dist = float('inf')
        best_id = -1
        
        for pdf_id, pdf_centroid in pdf_ids.items():
            dist = math.sqrt((obj['centroid_2d'][0] - pdf_centroid[0])**2 + (obj['centroid_2d'][1] - pdf_centroid[1])**2)
            
            if dist < min_dist and dist < 50:
                min_dist = dist
                best_id = pdf_id
        
        if best_id != -1 and final_objects[best_id] is None:
             obj['id'] = best_id
             final_objects[best_id] = obj

    return final_objects


def draw_results_3d(image, objects):
    output_image = image.copy()
    print("--- Part C: 3D Position Estimation Results ---")
    
    for obj in objects:
        if obj is None:
            continue
            
        i = obj['id']
        u, v = obj['centroid_2d']
        X, Y, Z = obj['coord_3d']

        # Print to terminal
        text = f"ID {i}: Coord ({X:.2f}, {Y:.2f}, {Z:.2f})"
        print(text)
        
        # Draw ID text on the image
        cv2.putText(output_image, f"ID {i}", 
                    (int(u)-20, int(v)-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 1)
    
    return output_image


def main():
    # --- Reading command line ---
    if len(sys.argv) != 3:
        print("Usage: python3 hw3_c.py <rgb_image_path> <depth_image_path>")
        print("Example: python3 hw3_c.py hw3/b/cubes.png hw3/c/depth.png")
        return
    
    # The first argument is the RGB image path, the second is the Depth image path.
    # From command line
    rgb_path = sys.argv[1]
    depth_path = sys.argv[2]
    
    # --- Step 1 : Load image ---
    # Load RGB image
    image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Cannot load RGB image {rgb_path}")
        return
        
    # Load Depth image
    # We must use IMREAD_UNCHANGED to read all channels of the image, including depth data.
    # Reference: https://blog.gtwang.org/programming/opencv-basic-image-read-and-write-tutorial/
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        print(f"Error: Cannot load Depth image {depth_path}")
        return
        
    # --- Step 2 : The definition of K matrix ---
    # This is the matrix provided in the PDF for Part C.
    K = np.array([
        [613.57, 0.0,    286.54],
        [0.0,    613.54, 251.36],
        [0.0,    0.0,    1.0   ]
    ])

    # Detection and 3D calculation, just like Part B.
    objects = find_objects_and_properties(image, depth_image, K)
    
    # Draw results
    output_image = draw_results_3d(image, objects)
    
    # Display the final image
    cv2.imshow("Part C - 3D Position Result (Press 'q' to exit)", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()