import hashlib
from ..models import Camera
from ..extensions import db


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
