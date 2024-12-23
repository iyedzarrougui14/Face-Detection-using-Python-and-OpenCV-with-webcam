import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np

# Load the pre-trained model
model = load_model('smnist.h5')

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# List of predicted letters
letter_pred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, c = frame.shape

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max, y_max = 0, 0
            x_min, y_min = w, h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_max = max(x_max, x)
                x_min = min(x_min, x)
                y_max = max(y_max, y)
                y_min = min(y_min, y)
            
            # Add padding to the bounding box
            y_min = max(0, y_min - 20)
            y_max = min(h, y_max + 20)
            x_min = max(0, x_min - 20)
            x_max = min(w, x_max + 20)

            # Extract the region of interest (ROI)
            roi = frame_rgb[y_min:y_max, x_min:x_max]
            roi_resized = cv2.resize(roi, (28, 28))
            roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2GRAY)
            roi_resized = roi_resized / 255.0
            roi_resized = np.reshape(roi_resized, (1, 28, 28, 1))

            # Make prediction
            prediction = model.predict(roi_resized)
            predicted_letter = letter_pred[np.argmax(prediction)]

            # Display the predicted letter on the frame
            cv2.putText(frame, predicted_letter, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Draw the bounding box and hand landmarks
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, handLMs, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Frame", frame)

    # Break the loop if the ESC key is pressed
    if cv2.waitKey(1) % 256 == 27:
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()
