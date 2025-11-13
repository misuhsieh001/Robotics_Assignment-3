import numpy as np
import cv2
import glob
import os
import sys

# ==============================================================================
# Calibration Settings
# ==============================================================================

# The number of inner corners per a chessboard row and column
CHECKERBOARD = (8, 6) 

# The real-world size of a square on the chessboard (in mm)
SQUARE_SIZE = 18.0  #Canon PowerShot
#SQUARE_SIZE = 20.0  #Iphone Blackmagic
# Set the termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# For storing object points and image points from all the images.
objpoints = [] # 3D point in real world space
imgpoints = [] # 2D points in image plane.

# Create a single object point grid for the chessboard pattern
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

# What mgid do is to create a grid of points for the chessboard corners in 3D space.
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE
print(objp)
images = glob.glob('calibration_images_2/*.JPG') #Canon PowerShot
#images = glob.glob('calibration_images/*.PNG')  #Iphone Blackmagic
print(images)
if len(images) == 0:
    print("--- Fatal Error ---")
    print("No images. Check 'calibration_images/' and file type is .JPG/.png")
    sys.exit()

print(f"Find {len(images)} images. Start detecting corner point...")
h, w = 0, 0 # Image height and width

# Set OpenCV window for visualization
WINDOW_NAME = 'Calibration Check'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 900, 700) 



detected_count = 0

for i, fname in enumerate(images):
    img = cv2.imread(fname)
    if img is None:
        print(f"Error：Cannot load image: {fname}，continue...")
        continue
    
    if h == 0 and w == 0:
        h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
    
    if ret == True:
        print(f"✅ Image {i+1}/{len(images)} ({os.path.basename(fname)})：successful。")
        detected_count += 1
        
        # Refine corner locations to sub-pixel accuracy
        # (11,11) is the search window size
        # (-1,-1) indicates that there is no zero zone in the search
        # Reference: https://blog.csdn.net/qq_30815237/article/details/87179830
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Append object points and image points
        objpoints.append(objp)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        img_drawn = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners2, ret)
        
        # Scale down for display
        scale_percent = 50 
        width_resized = int(img_drawn.shape[1] * scale_percent / 100)
        height_resized = int(img_drawn.shape[0] * scale_percent / 100)
        resized_img = cv2.resize(img_drawn, (width_resized, height_resized), interpolation = cv2.INTER_AREA)

        # Show the image with detected corners
        cv2.imshow(WINDOW_NAME, resized_img)
        cv2.setWindowTitle(WINDOW_NAME, f'SUCCESS ({detected_count}/{len(images)}) - {os.path.basename(fname)}')
        cv2.waitKey(500) 

    else:
        # Detection failed
        print(f"❌ Image {i+1}/{len(images)} ({os.path.basename(fname)})：Detection failed.")
        cv2.setWindowTitle(WINDOW_NAME, f'FAILURE - {os.path.basename(fname)}')
        # Scale down for display
        scale_percent = 50 
        width_resized = int(img.shape[1] * scale_percent / 100)
        height_resized = int(img.shape[0] * scale_percent / 100)
        resized_img_fail = cv2.resize(img, (width_resized, height_resized), interpolation = cv2.INTER_AREA)
        cv2.imshow(WINDOW_NAME, resized_img_fail)
        cv2.waitKey(500)
        
cv2.destroyAllWindows()

if detected_count < 5: # Need at least 5 images for reliable calibration
    print(f"\n--- Warning: There are only ({detected_count} ) images---")
elif detected_count == 0:
    print("\n--- Fatal error : Detection error ---")
    sys.exit()

print(f"\n--- Successfully detected {detected_count} images. Start calibration ---")

# Execute camera calibration
ret_calib, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

print("\n--- Calibration Result ---")
print("Intrinsic Matrix K :\n", mtx)
print("\nDistortion Coefficient D (dist) :\n", dist)

# Calculate re-projection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

total_mean_error = mean_error / len(objpoints)
print("\nMean Re-projection Error :", total_mean_error)


# ==============================================================================
# Image Undistortion Demonstration
# ==============================================================================

# Optimize the camera matrix based on free scaling parameter
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

print("\nOptimized Intrinsic Matrix (newcameramtx):\n", newcameramtx)

# Read the 20 calibration images and test image for undistortion demonstration
#test_image = "bottle.png" #Iphone Blackmagic
test_image = "bottle_2.jpg" # Canon PowerShot
images.append(test_image)
if len(images) > 0:
    for i, image in enumerate(images):
        original_img = cv2.imread(image)
        if original_img is not None:
            
            # Undistort the image
            undistorted_img = cv2.undistort(original_img, mtx, dist, None, newcameramtx)
            
            # Crop the image based on the ROI
            x, y, w_crop, h_crop = roi
            undistorted_img_cropped = undistorted_img[y:y+h_crop, x:x+w_crop]
            # Display the original and undistorted images side by side
            cv2.namedWindow('Undistortion Result', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Undistortion Result', 1200, 600)
            
            # Stack original and undistorted images side by side
            display_img = np.hstack((original_img, undistorted_img))

            cv2.imshow('Undistortion Result', display_img)
            cv2.setWindowTitle('Undistortion Result', f'Undistortion Result {i+1}/26')
            print("\nPress any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
print("\n--- Part A: Calibration Successful---")