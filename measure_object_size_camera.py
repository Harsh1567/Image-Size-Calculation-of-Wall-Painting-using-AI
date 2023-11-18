# import cv2
# from object_detector import *
# import numpy as np

# # Load Aruco detector
# parameters = cv2.aruco.DetectorParameters()
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)


# # Load Object Detector
# detector = HomogeneousBgDetector()

# # Load Cap
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# while True:
#     _, img = cap.read()

#     # Get Aruco marker
#     corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
#     if corners:

#         # Draw polygon around the marker
#         int_corners = np.int0(corners)
#         cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

#         # Aruco Perimeter
#         aruco_perimeter = cv2.arcLength(corners[0], True)

#         # Pixel to cm ratio
#         pixel_cm_ratio = aruco_perimeter / 20

#         contours = detector.detect_objects(img)

#         # Draw objects boundaries
#         for cnt in contours:
#             # Get rect
#             rect = cv2.minAreaRect(cnt)
#             (x, y), (w, h), angle = rect

#             # Get Width and Height of the Objects by applying the Ratio pixel to cm
#             object_width = w / pixel_cm_ratio
#             object_height = h / pixel_cm_ratio

#             # Display rectangle
#             box = cv2.boxPoints(rect)
#             box = np.int0(box)

#             cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
#             cv2.polylines(img, [box], True, (255, 0, 0), 2)
#             cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
#             cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)



#     cv2.imshow("Image", img)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# from object_detector import *
# import numpy as np

# # Load Aruco detector
# parameters = cv2.aruco.DetectorParameters()
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# # Load Object Detector
# detector = HomogeneousBgDetector()

# # Load image
# image_path = "5.jpeg"  # Replace with the path to your image file
# img = cv2.imread(image_path)

# # Get Aruco marker
# corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
# if corners:
#     # Draw polygon around the marker
#     int_corners = np.int0(corners)
#     cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

#     # Aruco Perimeter
#     aruco_perimeter = cv2.arcLength(corners[0], True)

#     # Pixel to cm ratio
#     pixel_cm_ratio = aruco_perimeter / 20

#     contours = detector.detect_objects(img)

#     # Draw objects boundaries
#     for cnt in contours:
#         # Get rect
#         rect = cv2.minAreaRect(cnt)
#         (x, y), (w, h), angle = rect

#         # Get Width and Height of the Objects by applying the Ratio pixel to cm
#         object_width = w / pixel_cm_ratio
#         object_height = h / pixel_cm_ratio

#         # Display rectangle
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)

#         cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
#         cv2.polylines(img, [box], True, (255, 0, 0), 2)
#         cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
#         cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# for all object detect with aruco in all dimension

# import cv2
# from object_detector import *
# import numpy as np

# # Load Aruco detector
# parameters = cv2.aruco.DetectorParameters()
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# # Load Object Detector
# class HomogeneousBgDetector:
#     def detect_objects(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # Create a Mask with adaptive threshold
#         mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

#         # Find contours
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         #cv2.imshow("mask", mask)
#         objects_contours = []

#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if area > 2000:
#                 #cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
#                 objects_contours.append(cnt)

#         return objects_contours
#         pass

# detector = HomogeneousBgDetector()

# # Set focal length of the camera (estimated or calibrated)
# focal_length = 200.0  # Adjust this value based on your camera

# # Calculate distance to marker based on marker size and focal length
# def calculate_distance(marker_width, focal_length, measured_width):
#     distance = (marker_width * focal_length) / measured_width
#     return distance

# # Open the webcam
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# while True:
#     # Read frame from the webcam
#     ret, frame = cap.read()

#     # Detect Aruco markers in the frame
#     corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

#     if ids is not None and len(ids) > 0:
#         for i in range(len(ids)):
#             # Get the corners of the detected marker
#             marker_corners = corners[i][0]

#             # Calculate the measured width of the marker in pixels
#             measured_width = np.linalg.norm(marker_corners[0] - marker_corners[1])

#             # Calculate the distance to the marker
#             distance = calculate_distance(10.0, focal_length, measured_width)

#             # Calculate dynamic marker width and height based on distance
#             marker_width = (focal_length) / distance
#             marker_height = marker_width

#             # Draw a rectangle around the marker and display the distance
#             cv2.aruco.drawDetectedMarkers(frame, [corners[i]])
#             cv2.putText(frame, "Distance: {:.2f} cm".format(distance), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             cv2.putText(frame, "Width: {:.2f} cm".format(marker_width), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             cv2.putText(frame, "Height: {:.2f} cm".format(marker_height), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


#      # Pixel to cm ratio
#     aruco_perimeter = cv2.arcLength(corners[i], True)
#     pixel_cm_ratio = aruco_perimeter / 20

#     # Detect other objects
#     contours = detector.detect_objects(frame)
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         object_width = w / pixel_cm_ratio
#         object_height = h / pixel_cm_ratio

#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.putText(frame, "Width: {:.2f} cm".format(object_width), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
#         cv2.putText(frame, "Height: {:.2f} cm".format(object_height), (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#     # Display the frame
#     cv2.imshow("Object Detection", frame)

#     # Check for the 'q' key to exit the loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close all windows
# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Load Aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# Set focal length of the camera (estimated or calibrated)
focal_length = 500.0  # Adjust this value based on your camera

# Calculate distance to marker based on marker size and focal length
def calculate_distance(marker_width, focal_length, measured_width):
    distance = (marker_width * focal_length) / measured_width
    return distance

# Load Object Detector
class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # Convert Image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a Mask with adaptive threshold
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                objects_contours.append(cnt)

        return objects_contours

# Create object detector instance
detector = HomogeneousBgDetector()

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Detect Aruco markers in the frame
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None and len(ids) > 0:
        # Get the corners of the first detected marker
        marker_corners = corners[0][0]

        # Calculate the measured width of the marker in pixels
        measured_width = np.linalg.norm(marker_corners[0] - marker_corners[1])

        # Calculate the distance to the marker
        distance = calculate_distance(4, focal_length, measured_width)

        # Calculate dynamic marker width and height based on distance
        marker_width = (focal_length) / distance
        marker_height = marker_width

        # Draw a rectangle around the marker and display the distance
        cv2.aruco.drawDetectedMarkers(frame, corners)
        cv2.putText(frame, "Distance: {:.2f} cm".format(distance), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Width: {:.2f} cm".format(marker_width), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Height: {:.2f} cm".format(marker_height), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Detect objects using object detector
        contours = detector.detect_objects(frame)

        # Draw objects boundaries
        for cnt in contours:
            # Get rect
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            # Get Width and Height of the Objects by applying the Ratio pixel tocm
            object_width = w / marker_width
            object_height = h / marker_height

            # Display rectangle
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.polylines(frame, [box], True, (255, 0, 0), 2)
            cv2.putText(frame, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            cv2.putText(frame, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

    # Display the frame
    cv2.imshow("Aruco Marker Distance", frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()



     

