import base64
from flask import Flask, jsonify, request
from dotenv  import load_dotenv
import cv2
import numpy as np
from imutils import contours
import imutils
import math
load_dotenv()
app = Flask(__name__)
omr_db = []
app.config['DEBUG_TB_INTERCEPT_REDIRECTS'] = False
print(cv2.__version__)
mtx = np.array([[816.95762135, 0., 499.10493361],
                [0., 817.0647965, 311.55296855],
                [0., 0., 1.]])
dist = np.array([[0.10622961, -1.0958522, 0.01852553, -0.01129245, 6.51470973]])
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
params = cv2.aruco.DetectorParameters_create()

def detect_ArUco(img):
    Detected_ArUco_markers = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    parameters = cv2.aruco.DetectorParameters()
    if corners:
        for ids, corners in zip(ids, corners):
            Detected_ArUco_markers.update({ids[0]: corners})

    return Detected_ArUco_markers

def degree1(Detected_ArUco_markers):
    ArUco_marker_angles = {}
    ## enter your code here ##
    for key in Detected_ArUco_markers:
        corners = Detected_ArUco_markers[key]
        tl = corners[0][0]  # top left
        tr = corners[0][1]  # top right
        br = corners[0][2]  # bottom right
        bl = corners[0][3]  # bottom left
        top = (tl[0] + tr[0]) / 2, -((tl[1] + tr[1]) / 2)
        centre = (tl[0] + tr[0] + bl[0] + br[0]) / 4, -(
            (tl[1] + tr[1] + bl[1] + br[1]) / 4
        )
        try:
            angle = round(
                math.degrees(np.arctan((top[1] - centre[1]) / (top[0] - centre[0])))
            )
        except:
            # add some conditions for 90 and 270
            if top[1] > centre[1]:
                angle = 90
            elif top[1] < centre[1]:
                angle = 270
        if top[0] >= centre[0] and top[1] < centre[1]:
            angle = 360 + angle
        elif top[0] < centre[0]:
            angle = 180 + angle
        ArUco_marker_angles.update({key: angle})
        print(ArUco_marker_angles)
        sum_values = sum(ArUco_marker_angles.values())
        new_Angle = sum_values/4 - 90
    return new_Angle  ## returning the angles of the ArUco markers in degrees as a dictionary


def degree(Detected_ArUco_markers):
    ArUco_marker_angles = {}
    for key in Detected_ArUco_markers:
        corners = Detected_ArUco_markers[key]
        tl = corners[0][0]  # top left
        tr = corners[0][1]  # top right
        br = corners[0][2]  # bottom right
        bl = corners[0][3]  # bottom left

        top = (tl[0] + tr[0]) / 2, -((tl[1] + tr[1]) / 2)
        centre = (tl[0] + tr[0] + bl[0] + br[0]) / 4, -((tl[1] + tr[1] + bl[1] + br[1]) / 4)

        try:
            angle = round(math.degrees(np.arctan((top[1] - centre[1]) / (top[0] - centre[0]))))
        except ZeroDivisionError:
            if top[1] > centre[1]:
                angle = 90
            elif top[1] < centre[1]:
                angle = 270
            else:
                angle = 0  # This handles the case where the marker is perfectly horizontal

        if top[0] >= centre[0] and top[1] < centre[1]:
            angle = 360 + angle
        elif top[0] < centre[0]:
            angle = 180 + angle

        ArUco_marker_angles[key] = angle
        print(f"Marker {key} angle: {angle}")

    sum_values = sum(ArUco_marker_angles.values())
    if len(ArUco_marker_angles) == 0:
        new_Angle = 0
    else:
        new_Angle = sum_values / len(ArUco_marker_angles) - 90
    return new_Angle

def adjust_angle(angle):
    # Calculate the remainder when the angle is divided by 90
    remainder = angle % 90

    # Determine the nearest multiple of 90
    if remainder <= 5 or remainder >= 85:
        # Adjust angle upwards or downwards to the nearest multiple of 90
        if remainder <= 5:
            adjusted_angle = angle - remainder
        else:
            adjusted_angle = angle + (90 - remainder)
    else:
        # If the remainder is not within the snapping range, keep the original angle
        adjusted_angle = angle

    return adjusted_angle



def rotate_image(array, angle):
   height, width = array.shape[:2]
   image_center = (width / 2, height / 2)

   rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

   radians = math.radians(angle)
   sin = math.sin(radians)
   cos = math.cos(radians)
   bound_w = int((height * abs(sin)) + (width * abs(cos)))
   bound_h = int((height * abs(cos)) + (width * abs(sin)))

   rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
   rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

   rotated_mat = cv2.warpAffine(array, rotation_mat, (bound_w, bound_h))
   return rotated_mat


# Function to detect Aruco markers in an image and find midpoints
def find_midpoints(image1):
    # Convert image to grayscale
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # Detect Aruco corners and IDs
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


    midpoints = []

    # Draw detected markers and corners
    if ids is not None:
        for i, corner in enumerate(corners):
            # Convert corners from float to integer for drawing
            int_corner = np.intp(corner.reshape(-1, 2))
            """
            # Draw marker boundary
            cv2.polylines(image, [int_corner], True, (0, 255, 0), 2)

            # Draw corner points
            for corner_point in int_corner:
                cv2.circle(image, tuple(corner_point), 5, (255,0,0), -1)
                """
            # Draw marker ID
            cv2.putText(image1, str(ids[i]), (int_corner[0, 0] - 3, int_corner[0, 1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

            # Calculate midpoints of the marker corners
            midpoint_x = int(np.mean(int_corner[:, 0]))
            midpoint_y = int(np.mean(int_corner[:, 1]))
            midpoint = (midpoint_x, midpoint_y)
            midpoints.append(midpoint)

            # Print ID, corner coordinates, and midpoint
            print("Marker ID:", ids[i])
            print("Midpoint:", midpoint)
            """
            # Draw midpoint
            cv2.circle(image, midpoint, 5, (0, 0, 255), -1)  """

    return image1, midpoints

# Function to find ROI from midpoints
def find_roi(image, midpoints):
    # Calculate the bounding box around midpoints
    min_x = min(midpoint[0] for midpoint in midpoints)
    max_x = max(midpoint[0] for midpoint in midpoints)
    min_y = min(midpoint[1] for midpoint in midpoints)
    max_y = max(midpoint[1] for midpoint in midpoints)

    # Define ROI coordinates
    roi_x = min_x
    roi_y = min_y
    roi_width = max_x - min_x
    roi_height = max_y - min_y

    # Extract ROI from the original image
    roi = image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    #roi = cv2.undistort(roi, mtx, dist, None, mtx)
    return roi


def perspective_transform(roi):
    # Define the four corners of the ROI
    height, width = roi.shape[:2]
    roi_corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # Define the desired ROI size after perspective transformation
    desired_size = (width, height)

    # Define the new ROI corners after perspective transformation
    new_roi_corners = np.float32([[0, 0], [desired_size[0], 0], [desired_size[0], desired_size[1]], [0, desired_size[1]]])

    # Compute the perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(roi_corners, new_roi_corners)

    # Apply the perspective transform
    transformed_roi = cv2.warpPerspective(roi, perspective_matrix, desired_size)

    return transformed_roi


def process(image):
    print(f"Original image shape: {image.shape}")

    # Ensure the image is in grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to grayscale")

    print(f"Grayscale image shape: {image.shape}")

    # Apply GaussianBlur to reduce image noise and improve thresholding
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    print("Applied GaussianBlur")

    # Apply Otsu's thresholding
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Applied Otsu's thresholding")

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 180
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 50
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    params.filterByCircularity = True
    params.minCircularity = 0.85
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(thresh)
    return keypoints

def bias(Average, h ):
    bi = abs(h-Average)
    print("bias",bi)
    return bi


def image_process(keypoints, transformed_roi, totalmcq, anskey, Columns):
    # Convert image to grayscale and apply blur
    gray = cv2.cvtColor(transformed_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge detection and thresholding
    edged = cv2.Canny(blurred, 75, 200)
    _, thresh = cv2.threshold(edged, 10, 255, cv2.THRESH_BINARY)

    # Draw keypoints
    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        cv2.circle(transformed_roi, (x, y), 4, (0, 0, 255), -1)
    # Find and sort contours
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnts = sorted(cnts[:Columns], key=lambda x: (cv2.boundingRect(x)[1], cv2.boundingRect(x)[0]))
    corners_list = []  # List to store corner coordinates of each contour
    sumx=0
    sumy=0
    for i, c in enumerate(cnts[:Columns]):
        x, y, w, h = cv2.boundingRect(c)
        top_left_corner = (x, y)
        sumx = sumx + x
        sumy = sumy + y
    Averagex= sumx/Columns
    Averagey= sumy/Columns
    print("sumx:",sumx,"sumy:",sumy,",Averagex:", Averagex,",Averagey:", Averagey,",Columns:",Columns)
    # Analyze and draw contours
    for i, c in enumerate(cnts[:Columns]):
        x, y, w, h = cv2.boundingRect(c)
        top_left_corner = (x, y)
        print(f"Contour {i+1}: x={x}, y={y}, width={w}, height={h}")
        cv2.rectangle(transformed_roi, (x, y), (x+w, y+h), ( 255, 0,0), 2)  # Draw bounding box
        # Print the top y-coordinate of each contour
        print(f"Top y-coordinate of Contour {i+1}: {y}")
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        corners_list.append(approx)
        if len(approx) == 4:
            x_diff = w
            y_diff = h
            secw = int(x_diff / totalmcq)
            sech = int(y_diff / len(anskey[i]))
            print(y)
            biy= bias(Averagey,y)
            for j in range(len(anskey[i])):
                corrAns = anskey[i][j]
                cx = (corrAns * secw) + secw // 2 + top_left_corner[0]
                cy = (j * sech) + sech // 2 + top_left_corner[1]
                print("coordinate of yellow is:", cx, cy)
                cv2.circle(transformed_roi, (cx, cy+ int(biy)), 6, (0, 255, 255), -1)
                for keypoint in keypoints:
                    blob_x = int(keypoint.pt[0])
                    blob_y = int(keypoint.pt[1])
                    if abs(cx  - blob_x) < 8 and abs(cy +biy - blob_y) < 8:
                        print("coordinate of green is:", cx, cy)
                        cv2.circle(transformed_roi, (cx, cy+int(biy)), 9, (0, 255, 0), -1)
                        break
        else:
            print("Contour", i+1, "does not have 4 corners")
    return transformed_roi


def count_red_blobs(processed_img):
    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 102, 102])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_blob_count = len(contours)
    return red_blob_count
def count_green_blobs(processed_img):
    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_blobs = len(contours)
    return green_blobs

# Retrieve all the OMRs
@app.route('/')
def get_all_omr():
    return jsonify({'OMR': omr_db})

# scan an omr
@app.route('/ScanOMR', methods=['POST'])
def scan_OMR():
    new_omr = request.get_json()
    nested_list = new_omr.get('nested_list')
    integer_value = new_omr.get('integer_value')
    Columns = new_omr.get('Columns')
    base64_image = new_omr.get('image_bytes')
    new_omr['nested_list'] = nested_list
    new_omr['integer_value'] = integer_value
    new_omr['image_bytes'] = base64_image
    new_omr['Columns'] = Columns
    image_bytes = base64.b64decode(base64_image)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    Detected_ArUco_markers = detect_ArUco(image)
    Angle = degree(Detected_ArUco_markers)
    print('Angle is : ', Angle)
    Angle = - adjust_angle(Angle)
    print('Updated Angle is : ', Angle)
    rotated_image = rotate_image(image, Angle)
    detected_image, midpoints= find_midpoints(rotated_image)
    roi = find_roi(rotated_image, midpoints)
    transformed_roi = perspective_transform(roi)
    keypoints= process(transformed_roi)
    processed_img = image_process(keypoints,transformed_roi,integer_value,nested_list,Columns)
    red_count = count_red_blobs(processed_img)
    print("Red Blob Count:", red_count)
    green_count = count_green_blobs(processed_img)
    print("green Blob Count:", green_count)
    # Convert processed image to base64
    _, processed_image_encoded = cv2.imencode('.jpg', processed_img)
    processed_image_base64 = base64.b64encode(processed_image_encoded).decode('utf-8')
    omr_db.append(new_omr)
    return jsonify({'correct_answers': green_count, 'wrong_answers': red_count, 'processed_image': processed_image_base64}),201
