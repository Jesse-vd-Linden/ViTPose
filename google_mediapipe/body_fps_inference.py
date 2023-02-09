import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


start = time.time()
start_inference = time.time()
experiment_duration = 60
inference_times = []
  
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while (time.time() - start < experiment_duration):
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS)
    
    # Flip the image horizontally for a selfie-view display.
    # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    # if cv2.waitKey(5) & 0xFF == 27:
    #   break

    inference_times.append(time.time() - start_inference)
    start_inference = time.time()

cap.release()


print(inference_times)
print("Inference time [ms]", np.average(inference_times)*1000)
print("FPS", (1 / np.average(inference_times)))