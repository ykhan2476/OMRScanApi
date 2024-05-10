import cv2
import numpy as np
from imutils import contours
import imutils

print(cv2.__version__)

mtx = np.array([[816.95762135, 0., 499.10493361],
                [0., 817.0647965, 311.55296855],
                [0., 0., 1.]])
dist = np.array([[0.10622961, -1.0958522, 0.01852553, -0.01129245, 6.51470973]])

# Define Aruco marker dictionary and parameters
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
parameters = cv2.aruco.DetectorParameters_create()
anskey=[[0,2,1,4,1,0,3,2,0,4,1,0,2,2],
        [0,2,1,4,1,0,3,2,0,4,1,0,2,2],
        [0,2,1,4,1,0,3,2,0,4,1,0,2,2],
        [0,2,1,4,1,0,3,2,0,4,1,0,2,2]]
totalmcq=5
def rvec_to_euler(rvec):
    rot=[]
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    theta_horizontal = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[2, 2])  # Angle with horizontal x-axis
    theta_horizontal_deg = np.degrees(theta_horizontal)
    print("Angle with horizontal x-axis:", theta_horizontal_deg)
    #eulerAngles_rad = cv2.RQDecomp3x3(rotation_matrix)[0]
    #eulerAngles_deg = np.degrees(eulerAngles_rad)

    #print("rot_mat:",rotation_matrix)
    #print("euler radians:",eulerAngles_rad)
    #print("euler degrees:",eulerAngles_deg)
    """
    # Extract Euler angles from rotation matrix
    theta_x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    theta_y = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    theta_z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])"""
    return  theta_horizontal_deg

# Function to detect Aruco markers in an image and find midpoints
def detect_aruco(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
    # Detect Aruco corners and IDs
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    
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
            cv2.putText(image, str(ids[i]), (int_corner[0, 0] - 3, int_corner[0, 1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,0), 2)
            
            # Calculate midpoints of the marker corners
            midpoint_x = int(np.mean(int_corner[:, 0]))
            midpoint_y = int(np.mean(int_corner[:, 1]))
            midpoint = (midpoint_x, midpoint_y)
            midpoints.append(midpoint)
            
            # Print ID, corner coordinates, and midpoint
            print("Marker ID:", ids[i])
            #print("Corner Coordinates:")
            #for j, coord in enumerate(corner):
                #print("Corner", j+1, ":", coord)
            print("Midpoint:", midpoint)
            """
            # Draw midpoint
            cv2.circle(image, midpoint, 5, (0, 0, 255), -1)  """
    
    #pose estimation 
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.02, mtx,dist)
    rotated=[]
    if len(corners) > 0:
        for i in range(0, len(rvec)):
            cv2.drawFrameAxes(image, mtx, dist, rvec[i], tvec[i], 0.01)
            print("Orientation (rvec):", rvec[i])
            rotated.append(rvec_to_euler(rvec[i])) 
    print("rotated:",rotated)
    cv2.imshow("Image with pose estimation", image)
    return image, midpoints


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
    
    return roi

def image_process(roi):
    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    # Apply thresholding to segment the image into binary regions
    _, thresh = cv2.threshold(edged, 10, 255, cv2.THRESH_BINARY)
    
    params = cv2.SimpleBlobDetector_Params()
    # Set up blob detector parameters
    params.minThreshold = 180
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 30
    params.filterByConvexity = True
    params.minConvexity = 0.9
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.5  
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs
    keypoints = detector.detect(gray)

    # Draw red circles around detected blobs
    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        cv2.circle(roi, (x, y), 4, (0, 0, 255), -1)
    
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
   
    corners_list = []  # List to store corner coordinates of each contour
    top_left_corners = []
    # Draw blue boundaries around contours and find corners
    for i, c in enumerate(cnts[:4]):
        cv2.drawContours(roi, [c], -1, (255, 0, 0), 2)  # Draw with blue color
        
        # Calculate contour area
        area = cv2.contourArea(c)
        print("Contour", i+1, "Area:", area)
        
        # Find the corners of the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        # Add corner coordinates to the list
        corners_list.append(approx)
        
        # Calculate the difference between x and y coordinates for each contour
        x_diff = approx[:, 0, 0].max() - approx[:, 0, 0].min()
        y_diff = approx[:, 0, 1].max() - approx[:, 0, 1].min()
        top_left_corner = tuple(c[c[:, :, 0].argmin()][0])
        top_left_corners.append(top_left_corner)
        print("Contour", i+1, "Top-Left Corner:", top_left_corner)
        print("Contour", i+1, "X Difference:", x_diff)
        print("Contour", i+1, "Y Difference:", y_diff)
        
        secw = int(x_diff / totalmcq)
        sech = int((y_diff) / len(anskey[0]))
        
        for j in range(0, len(anskey[0])):
            corrAns = anskey[i][j]
            cx = (corrAns * secw) + secw // 2 + top_left_corner[0]
            cy = (j * sech) + sech // 2 + top_left_corners[0][1]
            
            # Check if the green dot coincides with any detected blob
            for keypoint in keypoints:
                blob_x = int(keypoint.pt[0])
                blob_y = int(keypoint.pt[1])
                # If the green dot overlaps with a detected blob, color it green
                if abs(cx - blob_x) < 8 and abs(cy - blob_y) < 8:
                    cv2.circle(roi, (cx, cy), 9, (0, 255, 0), -1)
                    break  # Stop checking for this answer key coordinate once it's colored
                
    return roi
def count_red_blobs(processed_img):
    # Convert processed image to HSV color space
    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for red color in HSV
    lower_red = np.array([0, 102, 102])
    upper_red = np.array([10, 255, 255])
    
    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count the number of red blobs
    red_blob_count = len(contours)
    #cv2.imshow("Processed Image with red blobs", hsv)
    return red_blob_count
def count_green_blobs(processed_img):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of green blobs
    green_blobs = len(contours)
    #cv2.imshow("Processed Image with green blobs", hsv)
    return green_blobs

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

# Load an image
image = cv2.imread('im2.png')
#image = cv2.imread('omr3.jpg')
# Detect Aruco markers in the image and find midpoints
detected_image, midpoints = detect_aruco(image)
cv2.namedWindow("Image with Aruco Markers and Corners", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image with Aruco Markers and Corners", 600, 800)  # Set your desired width and height

# Display the image with detected markers and corners
cv2.imshow("Image with Aruco Markers and Corners",image)


# Find ROI based on midpoints
roi = find_roi(detected_image, midpoints)
"""
# Display the ROI
cv2.imshow("ROI", roi)"""
# Apply perspective transform on the ROI
transformed_roi = perspective_transform(roi)
cv2.namedWindow("Transformed ROI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Transformed ROI", 400, 400)  # Set your desired width and height
cv2.imshow("Transformed ROI", transformed_roi)
# Process the ROI
processed_img = image_process(transformed_roi)

# Display the processed image with blue contour boundaries
cv2.namedWindow("Processed Image with Blue Contours", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Processed Image with Blue Contours", 650, 700)  # Set your desired width and height
cv2.imshow("Processed Image with Blue Contours", processed_img)
# Example usage:
red_count = count_red_blobs(processed_img)
print("Red Blob Count:", red_count)
green_count = count_green_blobs(processed_img)
print("green Blob Count:", green_count)
  
cv2.waitKey(0)
cv2.destroyAllWindows()

#im5.png = area = 500
#im4.png= 2650
#im3.png = 100