import argparse
import threading
from faserver import app
from faserver.faceid.logic import face_recognition

if __name__ == "__main__":
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())

    # start a thread that will perform face identification
    t1 = threading.Thread(target=face_recognition)
    t1.daemon = True
    t1.start()

    # start flask app
    app.run(host=args["ip"], port=args["port"],
            debug=True, threaded=True, use_reloader=False)
