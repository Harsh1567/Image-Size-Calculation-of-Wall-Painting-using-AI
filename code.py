# import cv2
# import numpy as np
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse

# app = FastAPI()

# # Load Aruco detector
# parameters = cv2.aruco.DetectorParameters()
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# # Set focal length of the camera (estimated or calibrated)
# focal_length = 500.0  # Adjust this value based on your camera

# # Calculate distance to marker based on marker size and focal length
# def calculate_distance(marker_width, focal_length, measured_width):
#     distance = (marker_width * focal_length) / measured_width
#     return distance

# # Load Object Detector
# class HomogeneousBgDetector():
#     def __init__(self):
#         pass

#     def detect_objects(self, frame):
#         # Convert Image to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Create a Mask with adaptive threshold
#         mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

#         # Find contours
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         objects_contours = []

#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if area > 5000:
#                 objects_contours.append(cnt)

#         return objects_contours

# # Create object detector instance
# detector = HomogeneousBgDetector()

# # Open the webcam
# cap = cv2.VideoCapture(0)

# def video_stream():
#     while True:
#         # Read frame from the webcam
#         ret, frame = cap.read()

#         # Detect Aruco markers in the frame
#         corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

#         if ids is not None and len(ids) > 0:
#             # Get the corners of the first detected marker
#             marker_corners = corners[0][0]

#             # Calculate the measured width of the marker in pixels
#             measured_width = np.linalg.norm(marker_corners[0] - marker_corners[1])

#             # Calculate the distance to the marker
#             distance = calculate_distance(10.0, focal_length, measured_width)

#             # Calculate dynamic marker width and height based on distance
#             marker_width = (focal_length) / distance
#             marker_height = marker_width

#             # Draw a rectangle around the marker and display the distance
#             cv2.aruco.drawDetectedMarkers(frame, corners)
#             cv2.putText(frame, "Distance: {:.2f} cm".format(distance), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#             # Detect objects using object detector
#             contours = detector.detect_objects(frame)

#             # Draw objects boundaries
#             for cnt in contours:
#                 # Get the rect
#                 rect = cv2.minAreaRect(cnt)
#                 (x, y), (w, h), angle = rect
#                 object_width = w / marker_width
#                 object_height = h / marker_height

#                 # Display rectangle
#                 box = cv2.boxPoints(rect)
#                 box = np.intp(box)

#                 cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
#                 cv2.polylines(frame, [box], True, (255, 0, 0), 2)
#                 cv2.putText(frame, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
#                 cv2.putText(frame, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

#                 # Save the detected objects as images
#                 object_image = frame[int(y):int(y + h), int(x):int(x + w)]
#                 cv2.imwrite("detected_object.jpg", object_image)

#                 # Save the frame with the ArUco marker and objects
#                 cv2.imwrite("detected_frame.jpg", frame)

#                 # Load the saved image and display it in a separate window
#                 saved_frame = cv2.imread("detected_frame.jpg")
#                 cv2.imshow("Saved Frame", saved_frame)

#         # Convert frame to JPEG format
#         _, jpeg = cv2.imencode('.jpg', frame)

#         # Generate streaming response with the frame in JPEG format
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

#         # Check for the 'q' key to exit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the webcam and close all windows
#     cap.release()
#     cv2.destroyAllWindows()

# @app.get("/video_feed")
# async def get_video_feed():
#     return StreamingResponse(video_stream(), media_type="multipart/x-mixed-replace;boundary=frame")

# if __name__ == "__main__":
#     import nest_asyncio
#     nest_asyncio.apply()
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)


# import cv2
# import numpy as np
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse

# app = FastAPI()

# # Load Aruco detector
# parameters = cv2.aruco.DetectorParameters()
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# # Set focal length of the camera (estimated or calibrated)
# focal_length = 500.0  # Adjust this value based on your camera

# # Calculate distance to marker based on marker size and focal length
# def calculate_distance(marker_width, focal_length, measured_width):
#     distance = (marker_width * focal_length) / measured_width
#     return distance

# # Load Object Detector
# class HomogeneousBgDetector():
#     def __init__(self):
#         pass

#     def detect_objects(self, frame):
#         # Convert Image to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Create a Mask with adaptive threshold
#         mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

#         # Find contours
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         objects_contours = []

#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if area > 5000:
#                 objects_contours.append(cnt)

#         return objects_contours

# # Create object detector instance
# detector = HomogeneousBgDetector()

# # Open the webcam
# cap = cv2.VideoCapture(0)
# is_camera_on = True

# def video_stream():
#     global is_camera_on

#     while is_camera_on:
#         # Read frame from the webcam
#         ret, frame = cap.read()

#         # Detect Aruco markers in the frame
#         corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

#         if ids is not None and len(ids) > 0:
#             # Get the corners of the first detected marker
#             marker_corners = corners[0][0]

#             # Calculate the measured width of the marker in pixels
#             measured_width = np.linalg.norm(marker_corners[0] - marker_corners[1])

#             # Calculate the distance to the marker
#             distance = calculate_distance(10.0, focal_length, measured_width)

#             # Calculate dynamic marker width and height based on distance
#             marker_width = (focal_length) / distance
#             marker_height = marker_width

#             # Draw a rectangle around the marker and display the distance
#             cv2.aruco.drawDetectedMarkers(frame, corners)
#             cv2.putText(frame, "Distance: {:.2f} cm".format(distance), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#             # Detect objects using object detector
#             contours = detector.detect_objects(frame)

#             # Draw objects boundaries
#             for cnt in contours:
#                 # Get the rect
#                 rect = cv2.minAreaRect(cnt)
#                 (x, y), (w, h), angle = rect
#                 object_width = w / marker_width
#                 object_height = h / marker_height

#                 # Display rectangle
#                 box = cv2.boxPoints(rect)
#                 box = np.intp(box)

#                 cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
#                 cv2.polylines(frame, [box], True, (255, 0, 0), 2)
#                 cv2.putText(frame, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
#                 cv2.putText(frame, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

#                 # Save the detected objects as images
#                 object_image = frame[int(y):int(y + h), int(x):int(x + w)]
#                 cv2.imwrite("detected_object.jpg", object_image)

#                 # Save the frame with the ArUco marker and objects
#                 cv2.imwrite("detected_frame.jpg", frame)

#                 # Load the saved image and display it in a separate window
#                 saved_frame = cv2.imread("detected_frame.jpg")
#                 cv2.imshow("Saved Frame", saved_frame)

#                 # Turn off the camera after capturing the image
#                 is_camera_on = False

#         # Convert frame to JPEG format
#         _, jpeg = cv2.imencode('.jpg', frame)

#         # Generate streaming response with the frame in JPEG format
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

#         # Check for the 'q' key to exit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the webcam and close all windows
#     cap.release()
#     cv2.destroyAllWindows()

# @app.get("/video_feed")
# async def get_video_feed():
#     return StreamingResponse(video_stream(), media_type="multipart/x-mixed-replace;boundary=frame")

# if __name__ == "__main__":
#     import nest_asyncio
#     nest_asyncio.apply()
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

# import cv2
# import numpy as np
# import base64
# import tempfile
# import datetime
# from fastapi import FastAPI, Form

# app = FastAPI()

# # Load Aruco detector
# parameters = cv2.aruco.DetectorParameters()
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# # Set focal length of the camera (estimated or calibrated)
# focal_length = 500.0  # Adjust this value based on your camera

# # Calculate distance to marker based on marker size and focal length
# def calculate_distance(marker_width, focal_length, measured_width):
#     distance = (marker_width * focal_length) / measured_width
#     return distance

# # Load Object Detector
# class HomogeneousBgDetector():
#     def __init__(self):
#         pass

#     def detect_objects(self, frame):
#         # Convert Image to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Create a Mask with adaptive threshold
#         mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

#         # Find contours
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         objects_contours = []

#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if area > 5000:
#                 objects_contours.append(cnt)

#         return objects_contours

# # Create object detector instance
# detector = HomogeneousBgDetector()

# def process_video(base64_video_data):
#     try:
#         # Decode base64 video data
#         video_data = base64.b64decode(base64_video_data)
#     except base64.binascii.Error:
#         raise ValueError("Invalid base64 video data")

#     # Create a temporary file and write the video data to it
#     with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
#         temp_file.write(video_data)
#         temp_file.flush()

#         # Create a VideoCapture object to read the video stream from the temporary file
#         cap = cv2.VideoCapture(temp_file.name)
#         frame_count = 0
#         captured_image = None

#         while True:
#             # Read frame from the video stream
#             ret, frame = cap.read()

#             if not ret:
#                 break

#             frame_count += 1

#             # Detect Aruco markers in the frame
#             corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

#             if ids is not None and len(ids) > 0:
#                 # Get the corners of the first detected marker
#                 marker_corners = corners[0][0]

#                 # Calculate the measured width of the marker in pixels
#                 measured_width = np.linalg.norm(marker_corners[0] - marker_corners[1])

#                 # Calculate the distance to the marker
#                 distance = calculate_distance(10.0, focal_length, measured_width)

#                 # Calculate dynamic marker width and height based on distance
#                 marker_width = (focal_length) / distance
#                 marker_height = marker_width

#                 # Draw a rectangle around the marker and display the distance
#                 cv2.aruco.drawDetectedMarkers(frame, corners)

#                 # Detect objects using object detector
#                 contours = detector.detect_objects(frame)

#                 # Draw objects boundaries
#                 for cnt in contours:
#                     # Get the rect
#                     rect = cv2.minAreaRect(cnt)
#                     (x, y), (w, h), angle = rect
#                     object_width = w / marker_width
#                     object_height = h / marker_height

#                     # Display rectangle
#                     box = cv2.boxPoints(rect)
#                     box = np.intp(box)

#                     cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
#                     cv2.polylines(frame, [box], True, (255, 0, 0), 2)
#                     cv2.putText(frame, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
#                     cv2.putText(frame, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

#                 if ids is not None and len(ids) > 0 and len(contours) > 0:
#                     # Save the frame as an image
#                     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#                     image_name = f"image_{timestamp}.jpg"
#                     cv2.imwrite(image_name, frame)
#                     captured_image = frame
#                     break

#         # Release the video capture object
#         cap.release()

#         return captured_image

# @app.post("/process_video/")
# async def get_processed_video(base64_video_data: str = Form(...)):
#     captured_image = process_video(base64_video_data)

#     # Encode the captured image to base64 JPEG format
#     if captured_image is not None:
#         _, img_encoded = cv2.imencode(".jpg", captured_image)
#         img_base64 = base64.b64encode(img_encoded).decode()

#         # Print the encoded image data in the console
#         print("Encoded Image Data:")
#         print(img_base64)

#     return {"success": True}

# if __name__ == "__main__":
#     import nest_asyncio
#     nest_asyncio.apply()
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

import cv2
import numpy as np
import base64
import tempfile
import datetime
from fastapi import FastAPI, Form, File
from fastapi.responses import StreamingResponse

app = FastAPI()

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
            if area > 5000:
                objects_contours.append(cnt)

        return objects_contours

# Create object detector instance
detector = HomogeneousBgDetector()

# Base64-encoded video data

def process_video(base64_video_data):
    try:
        # Decode base64 video data
        video_data = base64.b64decode(base64_video_data)
    except base64.binascii.Error:
        raise ValueError("Invalid base64 video data")

    # Create a temporary file and write the video data to it
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
        temp_file.write(video_data)
        temp_file.flush()

        # Create a VideoCapture object to read the video stream from the temporary file
        cap = cv2.VideoCapture(temp_file.name)
        frame_count = 0
        captured_image = None

        while True:
            # Read frame from the video stream
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Detect Aruco markers in the frame
            corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)


            if ids is not None and len(ids) > 0:
                # Get the corners of the first detected marker
                marker_corners = corners[0][0]

                # Calculate the measured width of the marker in pixels
                measured_width = np.linalg.norm(marker_corners[0] - marker_corners[1])

                # Calculate the distance to the marker
                distance = calculate_distance(10.0, focal_length, measured_width)

                # Calculate dynamic marker width and height based on distance
                marker_width = (focal_length) / distance
                marker_height = marker_width

                # Draw a rectangle around the marker and display the distance
                cv2.aruco.drawDetectedMarkers(frame, corners)

                # Detect objects using object detector
                contours = detector.detect_objects(frame)

                # Draw objects boundaries
                for cnt in contours:
                    # Get the rect
                    rect = cv2.minAreaRect(cnt)
                    (x, y), (w, h), angle = rect
                    object_width = w / marker_width
                    object_height = h / marker_height

                    # Display rectangle
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)

                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.polylines(frame, [box], True, (255, 0, 0), 2)
                    cv2.putText(frame, "Distance: {:.2f} cm".format(distance), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
                    cv2.putText(frame, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

                if ids is not None and len(ids) > 0 and len(contours) > 0:
                    # Save the frame as an image
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_name = f"image_{timestamp}.jpg"
                    cv2.imwrite(image_name, frame)
                    captured_image = frame
                    break

        # Release the video capture object
        cap.release()

        return captured_image

@app.post("/process_video/")
async def get_processed_video(base64_video_data: str = Form(...)):
    captured_image = process_video(base64_video_data)

    # Convert the captured image to base64-encoded JPEG format
    if captured_image is not None:
        _, img_encoded = cv2.imencode(".jpg", captured_image)
        img_base64 = base64.b64encode(img_encoded).decode()

        print(img_base64)

        # Display the captured image in a separate window
        # cv2.imshow('Captured Image', captured_image)
        # cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        # cv2.destroyAllWindows()

        return {"image": img_base64}
    else:
        return {"error": "No image captured"}

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



