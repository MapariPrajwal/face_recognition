import cv2
import pickle
import face_recognition as FR
import time
from collections import defaultdict

width = 640
height = 360

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

font = cv2.FONT_HERSHEY_SIMPLEX

# Load known face encodings and names
with open('train.pkl', 'rb') as f:
    names = pickle.load(f)
    knownEncodings = pickle.load(f)

head_tracker = defaultdict(lambda: {'engagement_start_time': None, 'total_engagement_time': 0})

while True:
    ignore, unknownFace = cam.read()

    unknownFaceRGB = cv2.cvtColor(unknownFace, cv2.COLOR_BGR2RGB)
    faceLocations = FR.face_locations(unknownFaceRGB)
    unknownEncodings = FR.face_encodings(unknownFaceRGB, faceLocations)

    for faceLocation, unknownEncoding in zip(faceLocations, unknownEncodings):
        top, right, bottom, left = faceLocation
        cv2.rectangle(unknownFace, (left, top), (right, bottom), (255, 255, 127), 2)
        name = 'Unknown Person'
        matches = FR.compare_faces(knownEncodings, unknownEncoding)
        if True in matches:
            matchIndex = matches.index(True)
            name = names[matchIndex]
        cv2.putText(unknownFace, name, (left, top), font, 0.75, (255, 127, 255), 2)

        # Update engagement time
        head_id = (top, right, bottom, left)
        if head_id in head_tracker:
            if head_tracker[head_id]['engagement_start_time'] is None:
                head_tracker[head_id]['engagement_start_time'] = time.time()
            head_tracker[head_id]['total_engagement_time'] = time.time() - head_tracker[head_id]['engagement_start_time']

    # Display head count and engagement time
    head_count = len(head_tracker)
    y_position = 30
    for i, (head_id, data) in enumerate(head_tracker.items()):
        y_position += 30
        cv2.putText(unknownFace, f'Head {i + 1} - Engagement Time: {int(data["total_engagement_time"])}s', (10, y_position), font, 0.75, (255, 255, 255), 2)

    cv2.putText(unknownFace, f'Total Head Count: {head_count}', (10, y_position + 30), font, 0.75, (255, 255, 255), 2)

    cv2.imshow('Frame', unknownFace)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()