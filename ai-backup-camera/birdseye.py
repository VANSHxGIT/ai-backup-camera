import cv2
import numpy as np
# Load image
image = cv2.imread(r"A:\E\College stuffs\Summer Time Coding saga\SummerTime Builds\ai-backup-camera\road.jpg")# Resize for consistency
image = cv2.resize(image, (800, 600))
# Define source points (corners of road)
src_points = np.float32([
    [300, 400],   # Bottom-left
    [500, 400],   # Bottom-right
    [700, 550],   # Top-right
    [100, 550]    # Top-left
])
# Define destination points (bird-eye view)
dst_points = np.float32([
    [200, 0],
    [600, 0],
    [600, 600],
    [200, 600]
])
# Compute perspective transform matrix - Homography
matrix = cv2.getPerspectiveTransform(src_points, dst_points)
# Apply warp
warped = cv2.warpPerspective(image, matrix, (800, 600))
# Draw selected points on original image
for point in src_points:
    cv2.circle(image, tuple(point.astype(int)), 8, (0, 0, 255), -1)
# Show results
cv2.imshow("Original Image", image)
cv2.imshow("Bird-Eye View", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()