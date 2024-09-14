from flask import Flask, Response
from flask_cors import CORS
import cv2
import numpy as np
import time
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cvlib as cv

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the pre-trained gender detection model
model = load_model('gender_detection.model.h5')

# Open webcam
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set resolution width
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set resolution height
time.sleep(2)  # Allow the camera to warm up

# Global variables for gender counts
male_count = 0
female_count = 0

# Define classes
classes = ['man', 'woman']

def detect_gender(frame):
    global male_count, female_count

    # Apply face detection
    faces, confidence = cv.detect_face(frame)

    # Reset gender counts for each frame
    male_count = 0
    female_count = 0

    # Initialize gender distribution
    gender_distribution = "No faces detected"

    # Loop through detected faces
    for idx, f in enumerate(faces):
        # Get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw rectangle around the face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        # Skip small detected faces
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Predict gender
        conf = model.predict(face_crop)[0]

        # Get label with max confidence
        idx = np.argmax(conf)
        label = classes[idx]

        # Update gender count
        if label == 'man':
            male_count += 1
        else:
            female_count += 1

        # Format the label with confidence score
        label_text = "{}: {:.2f}%".format(label, conf[idx] * 100)

        # Set label position
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Display label and confidence on frame
        cv2.putText(frame, label_text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Gender distribution analysis
    if male_count > female_count:
        gender_distribution = "More males than females"
    elif female_count > male_count:
        gender_distribution = "More females than males"
    else:
        gender_distribution = "Equal number of males and females"

    # Display male and female counts on the frame
    cv2.putText(frame, f'Males: {male_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f'Females: {female_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, gender_distribution, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame

def gen():
    """
    Generator function to stream video frames
    """
    while True:
        ret, frame = webcam.read()
        if not ret:
            continue

        # Resize frame and run gender detection
        frame = cv2.resize(frame, (640, 480))
        frame = detect_gender(frame)

        # Encode the frame for streaming
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame = jpeg.tobytes()

        # Stream the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    """
    Route to serve the live video feed
    """
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
