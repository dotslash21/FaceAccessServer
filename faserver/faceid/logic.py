import sys
import cv2
import time
import datetime
import threading
from .. import app
import imutils
from imutils.video import VideoStream
from ..utils.face_id import FaceIdentifier


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing the stream)
outputFrame = None
detection_result = None
lock = threading.Lock()

# initialize the video stream and allow the camera sensor to
# warmup
vs = VideoStream(src=0).start()
time.sleep(2.0)


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, detection_result, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


def face_recognition():
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, detection_result, lock

    # Initialize the FaceIdentifier class
    face_id = FaceIdentifier(
        app.config['ALIGNED_IMG_DB'],
        app.config['MTCNN_MODEL_DIR'],
        app.config['FACENET_PRETRAINED_MODEL_PATH'],
        app.config['SVC_CLASSIFIER_SAVE_PATH']
    )

    # loop over frames from the video stream
    while True:
        # check if reloading SVC required
        if app.config['SVC_RELOAD']:
            face_id.load_svc()
            app.config['SVC_RELOAD'] = False

        # Grab the frame
        frame = vs.read()
        # Resize the frame
        frame = imutils.resize(frame, width=350)

        # Perform face identification
        id_result = face_id.identify(frame)
        # Catch any errors in identification
        if type(id_result) != type(int()):
            # Get the detection results and bounding box
            (faceID, bounding_box, detection_probability) = id_result
            # Draw the bounding box
            cv2.rectangle(frame, (bounding_box[0][0], bounding_box[0][1]), (
                bounding_box[0][2], bounding_box[0][3]), (0, 255, 0), 2)
            # Draw the faceId below bounding box
            text_x = bounding_box[0][0]
            text_y = bounding_box[0][3] + 20
            cv2.putText(frame, faceID, (text_x, text_y),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)
            # Set the detection result variable
            detection_result = {'name': faceID,
                                'probability': detection_probability}
        else:
            detection_result = None

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()
