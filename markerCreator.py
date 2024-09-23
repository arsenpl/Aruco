import cv2
import cv2.aruco as aruco
import numpy as np

# Define the dictionary and its parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
marker_size = 400

# Create a blank image
img = np.ones((marker_size, marker_size), dtype=np.uint8) * 255

# Draw a marker on the image
marker_id = 0
for marker_id in range(20):
    marker = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    cv2.imshow("ArUco Marker", marker)
    cv2.imwrite(f"markers/aruco_marker{marker_id}.png", marker)
    #cv2.waitKey(0)
    #break
'''
# Save the marker image
cv2.imwrite("aruco_marker.png", marker)

# Display the marker
cv2.imshow("ArUco Marker", marker)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''