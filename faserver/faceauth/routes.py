import io
import cv2
import base64
import numpy as np
from imageio import imread
from .logic import getToken, add_camera_entry, face_recognition
from flask import Blueprint, request, jsonify

mod = Blueprint("faceauth", __name__, template_folder="templates")


@mod.route("/register-client", methods=['POST'])
def camera_register():
    if request.method == 'POST':
        camera_name = request.json['clientId']
        camera_serial_num = request.json['serialNumber']
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
        encoded_img_list = request.json["imageList"]
        id_dict = {}

        for encoded_image in encoded_img_list:
            # Get rid of the meta info
            if ',' in encoded_image:
                encoded_image = encoded_image.split(',')[1]

            img = imread(io.BytesIO(base64.b64decode(encoded_image)))

            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Shape the image for consistency
            print("[INFO] Reshaping image... ", end="")
            h, w = frame.shape[0:2]
            maxDim = max([h, w])
            extraPadding = 0 if maxDim > 2000 else 2000 - maxDim
            dh = (maxDim - h) // 2 + extraPadding
            dw = (maxDim - w) // 2 + extraPadding
            frame = cv2.copyMakeBorder(frame.copy(), dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            print("Reshaping done!")
            print("[INFO] Resultant frame shape:", frame.shape)

            result = face_recognition(frame)

            if result['status'] == 'PASS':
                id = result['id']
                probability = result['probability']

                if id in id_dict:
                    id_dict['id']['probability_sum'] += probability
                    id_dict['id']['count'] += 1
                else:
                    id_dict[id] = {
                        'name': id.replace('_', " "),
                        'probability_sum': probability,
                        'count': 1
                    }


        finalId = None
        finalName = None
        finalProbability = 0
        for key, value in id_dict.items():
            if value['probability_sum'] / value['count'] > finalProbability:
                finalProbability = value['probability_sum'] / value['count']
                finalName = value['name']
                finalId = key

        # Sanity check
        if not finalId or not finalName or not finalProbability:
            return jsonify({
                'status': 'FAIL',
                'message': 'Oops! Something went wrong... :/'
            })

        return jsonify({
            'status': 'PASS',
            'id': finalId,
            'name': finalName,
            'probability': finalProbability
        })
    else:
        return jsonify({
            'status': 'FAIL',
            'message': 'Invalid request!'
        })
