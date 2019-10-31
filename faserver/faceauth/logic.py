import hashlib
from .. import app
from ..models import Camera
from ..extensions import db
from ..utils.face_id import FaceIdentifier


def getToken(string):
    encoded_string = string.encode()
    hash_object = hashlib.md5(encoded_string)
    return hash_object.hexdigest()


def add_camera_entry(camera_name, camera_serial_num, camera_token):
    try:
        camera = Camera(camera_name=camera_name,
                        camera_serial_num=camera_serial_num,
                        camera_token=camera_token)
        db.session.add(camera)
        db.session.commit()

        return 0
    except BaseException:
        raise Exception(
            '[ERROR] Problem encountered while adding the camera entry to database!')


def face_recognition(frame):
    # Initialize the FaceIdentifier class
    face_id = FaceIdentifier(
        app.config['ALIGNED_IMG_DB'],
        app.config['MTCNN_MODEL_DIR'],
        app.config['FACENET_PRETRAINED_MODEL_PATH'],
        app.config['SVC_CLASSIFIER_SAVE_PATH']
    )

    if app.config['SVC_RELOAD']:
        face_id.load_svc()
        app.config['SVC_RELOAD'] = False

    id_result = face_id.identify(frame)

    # Catch any errors in identification
    if not isinstance(id_result, type(int())):
        # Get the detection results and bounding box
        (faceID, _, detection_probability) = id_result
        # Set the detection result variable
        detection_result = {'status': 'PASS',
                            'id': faceID,
                            'probability': detection_probability}
    else:
        detection_result = {'status': 'FAIL'}

    return detection_result
