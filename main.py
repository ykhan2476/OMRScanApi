from flask import Flask, jsonify, request

app = Flask(__name__)

books_db = [
    {'name': 'Secret', 'price': 250},
    {'name': 'Deep work', 'price': 347}
]
app.config['DEBUG_TB_INTERCEPT_REDIRECTS'] = False

import cv2
import numpy as np
from imutils import contours
import imutils
print(cv2.__version__)
mtx = np.array([[816.95762135, 0., 499.10493361],
                [0., 817.0647965, 311.55296855],
                [0., 0., 1.]])
dist = np.array([[0.10622961, -1.0958522, 0.01852553, -0.01129245, 6.51470973]])
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
parameters = cv2.aruco.DetectorParameters_create()
anskey=[[0,2,1,4,1,0,3,2,0,4,1,0,2,2],
        [0,2,1,4,1,0,3,2,0,4,1,0,2,2],
        [0,2,1,4,1,0,3,2,0,4,1,0,2,2],
        [0,2,1,4,1,0,3,2,0,4,1,0,2,2]]
totalmcq=5
def rvec_to_euler(rvec):
    rot=[]
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    print("rot_mat:",rotation_matrix)
    theta_x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    theta_y = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    theta_z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    theta_x_deg = np.degrees(theta_x)
    theta_y_deg = np.degrees(theta_y)
    theta_z_deg = np.degrees(theta_z)
    rot.append(theta_x_deg)
    rot.append(theta_y_deg)
    rot.append(theta_z_deg)
    return rot

def detect_aruco(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    midpoints = []
    if ids is not None:
        for i, corner in enumerate(corners):
            int_corner = np.intp(corner.reshape(-1, 2))
            cv2.putText(image, str(ids[i]), (int_corner[0, 0] - 3, int_corner[0, 1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,0), 2)
            midpoint_x = int(np.mean(int_corner[:, 0]))
            midpoint_y = int(np.mean(int_corner[:, 1]))
            midpoint = (midpoint_x, midpoint_y)
            midpoints.append(midpoint)
            print("Marker ID:", ids[i])
            print("Midpoint:", midpoint)
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.02, mtx,dist)
    rotated=[]
    if len(corners) > 0:
        for i in range(0, len(rvec)):
            cv2.drawFrameAxes(image, mtx, dist, rvec[i], tvec[i], 0.01)
            print("Orientation (rvec):", rvec[i])
            rotated.append(rvec_to_euler(rvec[i]))  
    print("rotated:",rotated)
    return image, midpoints

def find_roi(image, midpoints):
    min_x = min(midpoint[0] for midpoint in midpoints)
    max_x = max(midpoint[0] for midpoint in midpoints)
    min_y = min(midpoint[1] for midpoint in midpoints)
    max_y = max(midpoint[1] for midpoint in midpoints)
    roi_x = min_x
    roi_y = min_y
    roi_width = max_x - min_x
    roi_height = max_y - min_y
    roi = image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    return roi

def image_process(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    _, thresh = cv2.threshold(edged, 10, 255, cv2.THRESH_BINARY)
    params = cv2.SimpleBlobDetector_Params()
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
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
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
        area = cv2.contourArea(c)
        print("Contour", i+1, "Area:", area)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        corners_list.append(approx)
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
            for keypoint in keypoints:
                blob_x = int(keypoint.pt[0])
                blob_y = int(keypoint.pt[1])
                if abs(cx - blob_x) < 8 and abs(cy - blob_y) < 8:
                    cv2.circle(roi, (cx, cy), 9, (0, 255, 0), -1)
                    break  
    return roi

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

def perspective_transform(roi):
    height, width = roi.shape[:2]
    roi_corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    desired_size = (width, height)
    new_roi_corners = np.float32([[0, 0], [desired_size[0], 0], [desired_size[0], desired_size[1]], [0, desired_size[1]]])
    perspective_matrix = cv2.getPerspectiveTransform(roi_corners, new_roi_corners)
    transformed_roi = cv2.warpPerspective(roi, perspective_matrix, desired_size) 
    return transformed_roi



# Retrieve all the books
@app.route('/books')
def get_all_books():
    return jsonify({'books': books_db})

# Retrieve all the books
@app.route('/books/images')
def get_all_booksimg():
    return "hey there how are you "

# Retrieve one book by name
@app.route('/book/<string:name>')
def get_book(name):
    for book in books_db:
        if book['name'] == name:
            return jsonify(book)
    return jsonify({'message': 'Book not found'}), 404

# Create a book
@app.route('/book', methods=['POST'])
def create_book():
    new_book = request.get_json()
    books_db.append(new_book)
    return jsonify({"message": "Book has been created"}), 201

# Update a book
@app.route('/book/<string:name>', methods=['PUT'])
def update_book(name):
    for book in books_db:
        if book['name'] == name:
            book_data = request.get_json()
            book.update(book_data)
            return jsonify({"message": "Book has been updated"})
    return jsonify({'message': 'Book not found'}), 404

# Delete a book
@app.route('/book/<string:name>', methods=['DELETE'])
def delete_book(name):
    for book in books_db:
        if book['name'] == name:
            books_db.remove(book)
            return jsonify({"message": "Book has been deleted"})
    return jsonify({'message': 'Book not found'}), 404


# Partially update a book (PATCH method)
@app.route('/book/<string:name>', methods=['PATCH'])
def partial_update_book(name):
    for book in books_db:
        if book['name'] == name:
            book_data = request.get_json()
            for key, value in book_data.items():
                book[key] = value
            return jsonify({"message": "Book has been partially updated"})
    return jsonify({'message': 'Book not found'}), 404

@app.route('/')
def get():
    image = cv2.imread('im2.png')
    detected_image, midpoints = detect_aruco(image)
    roi = find_roi(image, midpoints)
    transformed_roi = perspective_transform(roi)
    processed_img = image_process(transformed_roi)
    cv2.namedWindow("Processed Image with Blue Contours", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Processed Image with Blue Contours", 650, 700)  # Set your desired width and height
    cv2.imshow("Processed Image with Blue Contours", processed_img)
    red_count = count_red_blobs(processed_img)
    print("Red Blob Count:", red_count)
    green_count = count_green_blobs(processed_img)
    print("green Blob Count:", green_count)
    return {'correct answers': green_count,'wrong answers':red_count}
         
# Run the application
if __name__ == '__main__':
    app.run(port=5000,debug= True)
