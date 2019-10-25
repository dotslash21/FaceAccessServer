from .logic import getToken, add_camera_entry
from flask import Blueprint, request, jsonify

mod = Blueprint("faceauth", __name__, template_folder="templates")

@mod.route("/camera-register", methods=['POST'])
def camera_register():
    if request.method == 'POST':
        camera_name = request.data['name']
        camera_serial_num = request.data['serial_number']
        camera_token = getToken(camera_name + camera_serial_num)

        try:
            add_camera_entry(camera_name, camera_serial_num, camera_token)

            response_data = {
                'message': "Camera registered succesfully!",
                'data': {
                    'token': camera_token
                }
            }
            return jsonify(response_data)
        except:
            response_data = {
                'message': "There was a problem registering the camera"
            }
            return jsonify(response_data)
    else:
        response_data = {
            'message': "Invalid request!"
        }
        return jsonify(response_data)
