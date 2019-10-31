import base64
from .logic import getToken, add_camera_entry, face_recognition
from flask import Blueprint, request, jsonify

mod = Blueprint("faceauth", __name__, template_folder="templates")


@mod.route("/register-client", methods=['POST'])
def camera_register():
    if request.method == 'POST':
        camera_name = request.data['clientId']
        camera_serial_num = request.data['serialNumber']
        camera_token = getToken(camera_name + camera_serial_num)

        try:
            add_camera_entry(camera_name, camera_serial_num, camera_token)

            response_data = {
                'status': 'PASS',
                'message': "Camera registered succesfully!",
                'token': camera_token
            }
            return jsonify(response_data)
        except BaseException:
            response_data = {
                'status': 'FAIL',
                'message': "There was a problem registering the camera"
            }
            return jsonify(response_data)
    else:
        response_data = {
            'status': 'FAIL',
            'message': "Invalid request!"
        }
        return jsonify(response_data)


@mod.route("/face-auth", methods=['POST'])
def face_auth():
    if request.method == 'POST':
        encoded_img_list = list(request.data["imageList"])
        id_dict = {}

        for encoded_image in encoded_img_list:
            frame = base64.decodestring(encoded_image)

            result = face_recognition(frame)

            if result['status'] == 'PASS':
                id = result['id']
                probability = result['probability']

                if id in id_dict:
                    id_dict['id']['probability_sum'] += probability
                    id_dict['id']['count'] += 1
                else:
                    id_dict['id'] = {
                        'name': id.replace('_', " "),
                        'probability_sum': probability,
                        'count': 1
                    }

        finalId = ""
        finalName = ""
        finalProbability = 0
        for id, data in id_dict.items():
            if data['probability_sum'] / data['count'] > finalProbability:
                finalProbability = ['probability_sum'] / data['count']
                finalName = data['name']
                finalId = id

        return jsonify({
            'status': 'PASS',
            'id': finalId,
            'name': finalName,
            'probability': probability
        })
    else:
        return jsonify({
            'status': 'FAIL',
            'message': 'Invalid request!'
        })
