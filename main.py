import cv2 as cv
import numpy as np
from cv2 import aruco
import os


def image_augmentation (frame, src_image, dst_points):
    src_h, src_w = src_image.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    src_points = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_w]])
    H, _ = cv.findHomography(srcPoints=src_points, dstPoints=dst_points)
    warp_image = cv.warpPerspective(src_image, H, (frame_w, frame_h))
    #cv.imshow("warp", warp_image)
    cv.fillConvexPoly(mask, dst_points, 255)
    results=cv.bitwise_and(warp_image, warp_image, frame, mask=mask)

def read_images(dir_path):
    img_list = []
    files = os.listdir(dir_path)
    for file in files:
        img_path = os. path.join(dir_path, file)
        image = cv.imread(img_path)
        img_list.append(image)
    return img_list

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

param_markers = aruco.DetectorParameters()
cap = cv.VideoCapture("http://192.168.0.19:8080/video")
img_list=read_images("elements")
while True:
    ret, frame= cap.read()
    frame = cv.resize(frame, (960, 540))
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(gray_frame, aruco_dict, parameters=param_markers)

    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):
            #cv.polylines(frame, [corners.astype(np.int32)], True, (0,255,255), 4, cv.LINE_AA)
            #print(ids," ",corners)
            corners = corners.reshape(4,2)
            corners = corners.astype(int)

            image_augmentation(frame, img_list[ids[0]], corners)
            #top_right=corners[0].ravel()
            #cv.putText(frame, f"id: {ids[0]}", top_right, cv.FONT_HERSHEY_PLAIN, 1.3, (255,255,0), 2, cv.LINE_AA)

    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key==ord('q'):
        break
cap.release()
cv.destroyAllWindows()