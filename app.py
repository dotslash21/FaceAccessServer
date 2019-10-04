import os
import cv2
import sys
import time
import shutil
import imutils
import argparse
import threading
import datetime
from face_id import FaceIdentifier
from imutils.video import VideoStream
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, Response, render_template, url_for, request, redirect, flash

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
detection_result = None
lock = threading.Lock()

# Initialize flask object and configure it
app = Flask(__name__)
app.config['IMG_DB'] = 'img_db'
app.config['ALIGNED_IMG_DB'] = 'aligned_img_db'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# initialize the video stream and allow the camera sensor to
# warmup
vs = VideoStream(src=0).start()
time.sleep(2.0)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(40), nullable=False)
    last_name = db.Column(db.String(40), nullable=False)
    date_registered = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return "<USER-{0}>".format(self.id)


@app.route('/', methods=['GET'])
def index():
    users_count = User.query.count()
    return render_template('index.html', users_count=users_count)


@app.route('/add-user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        uploaded_files = request.files.getlist("images")

        try:
            user = User(first_name=first_name, last_name=last_name)
            db.session.add(user)
            db.session.commit()
        except:
            return '[ERROR] There was a problem adding the user!'

        print("[DEBUG] First name: {0} Last name: {1}".format(
            first_name, last_name), file=sys.stdout)

        for (index, file) in enumerate(uploaded_files):
            if file and allowed_file(file.filename) and index < 10:
                foldername = "{0}_{1}".format(first_name, last_name)
                directory = os.path.join(
                    app.config['IMG_DB'], foldername)
                extension = file.filename.rsplit('.', 1)[1].lower()
                filename = foldername + "_{0}.{1}".format(index, extension)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file.save(os.path.join(directory, filename))
                print("\'{0}\' has been saved at location \'{1}\'".format(
                    filename, directory), file=sys.stdout)

        os.system("python align_face.py")
        return redirect('/add-user')
    else:
        return render_template('add_user.html')


@app.route('/edit-user', methods=['GET', 'POST'])
def edit_user():
    if request.method == 'POST':
        id = request.form['user_id']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        old_first_name = request.form['old_first_name']
        old_last_name = request.form['old_last_name']
        uploaded_files = request.files.getlist("images")
        user = User.query.get_or_404(id)

        user.first_name = first_name
        user.last_name = last_name

        try:
            db.session.commit()
        except:
            return '[ERROR] There was a problem updating that task!'

        folder_name = "{0}_{1}".format(old_first_name, old_last_name)
        directory1 = os.path.join(app.config['IMG_DB'], folder_name)
        directory2 = os.path.join(app.config['ALIGNED_IMG_DB'], folder_name)
        shutil.rmtree(directory1)
        shutil.rmtree(directory2)

        for (index, file) in enumerate(uploaded_files):
            if file and allowed_file(file.filename) and index < 10:
                foldername = "{0}_{1}".format(first_name, last_name)
                directory = os.path.join(
                    app.config['IMG_DB'], foldername)
                extension = file.filename.rsplit('.', 1)[1].lower()
                filename = foldername + "_{0}.{1}".format(index, extension)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file.save(os.path.join(directory, filename))
                print("\'{0}\' has been saved at location \'{1}\'".format(
                    filename, directory), file=sys.stdout)

        os.system("python align_face.py")
        return redirect('/edit-user')
    else:
        users = User.query.order_by(User.date_registered).all()
        return render_template('edit_user.html', users=users)


@app.route('/delete-user', methods=['GET', 'POST'])
def delete_user():
    if request.method == 'POST':
        id = request.form['user_id']
        user = User.query.get_or_404(id)

        try:
            db.session.delete(user)
            db.session.commit()

            folder_name = "{0}_{1}".format(user.first_name, user.last_name)
            directory1 = os.path.join(app.config['IMG_DB'], folder_name)
            directory2 = os.path.join(
                app.config['ALIGNED_IMG_DB'], folder_name)
            shutil.rmtree(directory1)
            shutil.rmtree(directory2)

            os.system("python align_face.py")
            return redirect('/delete-user')
        except:
            return '[ERROR] There was a problem deleting that user!'
    else:
        users = User.query.order_by(User.date_registered).all()
        return render_template('delete_user.html', users=users)


@app.route('/view-feed', methods=['GET'])
def view_feed():
    global detection_result
    return render_template("view_feed.html", detection_result=detection_result)


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


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

    if vs == False:
        print("[ERROR] There was some problem opening camera video stream")
        sys.exit(1)

    face_id = FaceIdentifier()

    # loop over frames from the video stream
    while True:
        # Grab the frame
        frame = vs.read()
        # Resize the frame
        # frame = imutils.resize(frame, width=400)

        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

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


if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())

    # start a thread that will perform face identification
    t = threading.Thread(target=face_recognition)
    t.daemon = True
    t.start()

    # start flask app
    app.run(host=args["ip"], port=args["port"],
            debug=True, threaded=True, use_reloader=False)
