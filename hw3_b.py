import cv2
import sys
import numpy as np
import math

def find_objects_and_properties(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Image not found at {img_path}")
        return

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
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_image, lower, upper)
        
        if color_name == 'red1':
            mask_red2 = cv2.inRange(hsv_image, color_ranges['red2'][0], color_ranges['red2'][1])
            mask = cv2.bitwise_or(mask, mask_red2)
        elif color_name == 'red2':
            continue
        
        combined_mask = cv2.bitwise_or(combined_mask, mask)

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

    output_image = img.copy()
    
    print("--- Part B: Detection result ---")
    
    # Store detected results
    detected_results = [] 

    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 400: # Filter small contours, it's important to avoid noise.
            continue
            
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
            
        # Calculate centroid
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
        
        # Fit ellipse to get orientation
        if len(cnt) < 50:
            continue

        ellipse = cv2.fitEllipse(cnt)
        angle_deg = ellipse[2] # Angle in degrees(0-180)
        angle_rad = angle_deg * math.pi / 180.0
        
        detected_results.append({
            'id': -1, 
            'centroid': (cX, cY),
            'angle_deg': angle_deg if angle_deg <= 90 else angle_deg - 90, 
            'angle_rad': angle_rad
        })

    # Sort detected results by x-coordinate of centroid
    
    # PDF IDs and their expected centroid positions
    pdf_ids = {
        0: (170, 216), # Red
        1: (364, 208), # White
        2: (387, 91),  # Blue
        3: (163, 94),  # Green
        4: (245, 57)   # Yellow
    }

    final_objects = [None] * 5
    
    for obj in detected_results:
        min_dist = float('inf')
        best_id = -1
        
        for pdf_id, pdf_centroid in pdf_ids.items():
            dist = math.sqrt((obj['centroid'][0] - pdf_centroid[0])**2 + (obj['centroid'][1] - pdf_centroid[1])**2)
            
            if dist < min_dist and dist < 50:
                min_dist = dist
                best_id = pdf_id
        
        if best_id != -1 and final_objects[best_id] is None:
             obj['id'] = best_id
             final_objects[best_id] = obj
        else:
             pass 

    # Draw results
    for obj in final_objects:
        if obj is None:
            continue
            
        i = obj['id']
        cX, cY = obj['centroid']
        angle_deg = obj['angle_deg']
        angle_rad = obj['angle_rad']

        # Indicate centroid and orientation
        cv2.circle(output_image, (int(cX), int(cY)), 3, (0, 0, 255), -1)
        
        length = 100
        dx_std = int(length * math.cos(angle_rad))
        dy_std = int(length * math.sin(angle_rad))

        p1 = (int(cX - dx_std), int(cY - dy_std))
        p2 = (int(cX + dx_std), int(cY + dy_std))
        
        cv2.line(output_image, p1, p2, (255, 0, 0), 1)

        text = f"ID {i}: Centroid ({cX:.2f}, {cY:.2f}), Angle {angle_deg:.2f}"
        print(text)
        
        cv2.putText(output_image, f"ID {i}", 
                    (int(cX) - 20, int(cY) - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 255), 2)
        
    # Show final output
    while True:
        try:
            cv2.imshow("Part B Result (Fixed 13 - Press 'q' to exit)", output_image)
            cv2.imshow("Cleaned Mask (Debug)", cleaned_mask)
            key = cv2.waitKey(1)
            if key == ord("q") or key == 27:
                break
        except KeyboardInterrupt:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 hw3_b.py <input image>")
        sys.exit(1)
    find_objects_and_properties(sys.argv[1])