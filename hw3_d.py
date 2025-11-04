import cv2
import sys
import numpy as np
import math

# --- 步驟 1: 從 Part B 複製我們的偵測邏輯 ---
# (我們需要這個來 "找到" 方塊在哪裡)
def find_all_cubes_contours(img):
    """
    使用 Part B 的 HSV 邏輯來偵測 5 個方塊的輪廓。
    """
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # (使用我們在 Part B 最終除錯好的 HSV 範圍)
    color_ranges = {
        'red1': (np.array([0, 120, 70]), np.array([10, 255, 255])),
        'red2': (np.array([170, 120, 70]), np.array([180, 255, 255])),
        'green': (np.array([40, 100, 50]), np.array([80, 255, 255])),
        'blue': (np.array([90, 60, 30]), np.array([130, 255, 200])),
        'yellow': (np.array([20, 100, 100]), np.array([30, 255, 255])),
        'white_gray': (np.array([0, 0, 60]), np.array([180, 90, 255])),
    }
    
    combined_mask = np.zeros(img.shape[:2], dtype="uint8")

    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_image, lower, upper)
        if color_name == 'red1':
            mask_red2 = cv2.inRange(hsv_image, color_ranges['red2'][0], color_ranges['red2'][1])
            mask = cv2.bitwise_or(mask, mask_red2)
        elif color_name == 'red2':
            continue
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # (使用我們在 Part B 最終除錯好的兩階段清理)
    kernel_open = np.ones((3, 3), np.uint8)
    mask_opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
    kernel_close = np.ones((7, 7), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)
    
    # 找到清理乾淨的輪廓
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE) 
    
    # 過濾掉太小的
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    return valid_contours

# --- 步驟 2: 建立 3D 物件模型 ---
def create_cube_model(side_length=25.0):
    """
    建立一個 3D 方塊模型。我們必須假設一個物理尺寸。
    我們只定義 4 個頂面角點和 4 個底面角點。
    """
    s = side_length / 2.0
    # 我們定義 8 個角點
    # (注意：順序必須與 2D 偵測到的順序匹配)
    object_points = np.array([
        [-s, -s,  s], # 頂面 - 左後
        [ s, -s,  s], # 頂面 - 右後
        [ s,  s,  s], # 頂面 - 右前
        [-s,  s,  s], # 頂面 - 左前
        [-s, -s, -s], # 底面 - 左後
        [ s, -s, -s], # 底面 - 右後
        [ s,  s, -s], # 底面 - 右前
        [-s,  s, -s]  # 底面 - 左前
    ], dtype=np.float32)
    return object_points

# --- 步驟 3: 偵測 2D 影像點 (最困難的部分) ---
def find_corners_from_contour(contour):

    
    # 1. 找到輪廓的質心
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cX = M["m10"] / M["m00"]
    cY = M["m01"] / M["m00"]
    
    # 2. 逼近多邊形
    # 調整 epsilon (0.02 ~ 0.1) 是關鍵
    # Reference: https://vocus.cc/article/67ad8176fd897800019fcf64
    epsilon = 0.085 * cv2.arcLength(contour, True)
    approx_corners = cv2.approxPolyDP(contour, epsilon, True)
    
    # 3. 過濾：我們只尋找 4 到 7 個角點的方塊
    if len(approx_corners) < 4 or len(approx_corners) > 7:
        return None
        
    # 4. 排序 (極度困難)
    # PnP 演算法要求 2D 點的順序必須和 3D 模型的順序
    # (create_cube_model) 嚴格匹配。
    # 
    # 這裡我們只回傳偵測到的點，並假設它們的順序
    # 在 solvePnP 中是「部分正確」的 (這需要 PnP-RANSAC)
    
    # 我們只回傳 (N, 1, 2) 格式的角點
    return approx_corners.astype(np.float32)

# --- 步驟 4: 主函式 ---
def main():
    if len(sys.argv) != 2:
        print("Usage: python3 hw3_d.py <rgb_image_path>")
        print("Example: python3 hw3_d.py hw3/d/cubes.png")
        return

    rgb_path = sys.argv[1]
    
    K_matrix = np.array([
        [613.57, 0.0,    286.54],
        [0.0,    613.54, 251.36],
        [0.0,    0.0,    1.0   ]
    ])
    
    D_coeffs = np.zeros((4, 1), dtype=np.float32)

    # Assume the cube side length is 25 mm
    object_points_full_cube = create_cube_model(side_length=25.0)

    img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Cannot load RGB image {rgb_path}")
        return
    
    # 複製一份用於繪圖
    output_image = img.copy()

    # --- 執行偵測 ---
    all_contours = find_all_cubes_contours(img)
    
    print("--- Part D: 6D Pose Estimation Results (Bonus) ---")

    for contour in all_contours:
        # --- 步驟 3: 偵測 2D 角點 ---
        # 這是最困難且最不穩定的步驟
        image_points = find_corners_from_contour(contour)
        
        if image_points is None:
            continue
            
        # PnP 需要 3D 和 2D 點之間有對應
        # 由於我們不知道 8 個角點中有哪些是可見的，
        # 我們將只使用 3D 模型中的 4 個頂面角點
        object_points_top = object_points_full_cube[0:4]
        
        # 檢查我們是否剛好偵測到 4 個角點
        if len(image_points) != 4:
            # print("Warning: Did not find exactly 4 corners, skipping.")
            continue

        # --- 步驟 4: 執行 PnP ---
        try:
            # cv2.SOLVEPNP_ITERATIVE 是一種標準的 PnP 算法
            # 注意：solvePnP 非常依賴 2D/3D 點的 "順序"
            success, rvec, tvec = cv2.solvePnP(
                object_points_top,  # 3D 模型點
                image_points,       # 2D 影像點
                K_matrix,           # 內參 K
                D_coeffs            # 畸變 D
            )
            
            if success:
                # --- 步驟 5: 繪製結果 ---
                # tvec 是 3D 位置 (X, Y, Z)
                # rvec 是 3D 旋轉 (Roll, Pitch, Yaw)
                
                # `drawFrameAxes` 是一個很棒的函式，
                # 它會將 3D 模型的 X,Y,Z 軸畫在影像上
                # 讓我們看到 6D 姿態
                cv2.drawFrameAxes(
                    output_image, 
                    K_matrix, 
                    D_coeffs, 
                    rvec,  # 旋轉
                    tvec,  # 平移
                    20.0   # 座標軸的長度 (毫米)
                )
                
                # (您可以進一步使用 cv2.Rodrigues(rvec) 來獲取
                #  Roll, Pitch, Yaw 角度，但視覺化更有說服力)
                
        except cv2.error as e:
            print(f"cv2.solvePnP Error: {e}. Skipping cube.")


    # 顯示最終影像
    cv2.imshow("Part D - 6D Pose Result (Bonus)", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()