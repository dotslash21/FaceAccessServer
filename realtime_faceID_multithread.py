import os
import sys
import cv2
import pickle
import imutils
import argparse
import numpy as np
import tensorflow as tf
import utils.facenet as facenet
import utils.detect_face as detect_face
from PIL import Image
from imutils.video import FPS
from imutils.video import WebcamVideoStream

# Error Logging : FOR DEVELOPMENT ONLY!!
import logging
import traceback

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=30,
                help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
                help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

# Set allow_pickle=True
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

print('[INFO] Initializing networks and loading parameters...')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './npy')

        # Config variables
        minsize = 150  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        # Load the IDs form the db
        faceIds = os.listdir('./aligned_img_db/')
        faceIds.sort()

        # Load the facenet model and weights from file
        print('Loading feature extraction model...')
        modeldir = './pretrained_model/20170511-185253.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # Load the trained SVM Classifier from file
        classifier_filename = './my_classifier/SVC.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            print('[INFO] Loaded classifier file -> %s' %
                  classifier_filename_exp)

        # created a *threaded *video stream, allow the camera senor to warmup,
        # and start the FPS counter
        print("[INFO] sampling THREADED frames from webcam...")
        video_capture = WebcamVideoStream(src=0).start()

        # Check if video stream opened successfully
        if (video_capture == False):
            print("[ERROR] There was some problem opening camera video stream")
            sys.exit(1)

        # loop over some frames...this time using the threaded stream
        while True:
            # Grab the frame
            frame = video_capture.read()
            # Resize the frame
            frame = imutils.resize(frame, width=400)

            find_results = []

            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)

            frame = frame[:, :, 0:3]
            bounding_boxes, _ = detect_face.detect_face(
                frame, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            # print('Number of faces detected: %d' % nrof_faces)

            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(frame.shape)[0:2]

                cropped = []
                scaled = []
                scaled_reshape = []
                bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                for i in range(nrof_faces):
                    emb_array = np.zeros((1, embedding_size))

                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]

                    # inner exception
                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame[1]):
                        print('[ERROR] face is inner of range!')
                        continue

                    try:
                        cropped.append(
                            frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(np.array(Image.fromarray(cropped[i]).resize(
                            (image_size, image_size), resample=Image.BILINEAR)))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(
                            scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                        feed_dict = {
                            images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(
                            embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        # print(predictions)
                        best_class_indices = np.argmax(predictions, axis=1)
                        # print(best_class_indices)
                        best_class_probabilities = predictions[np.arange(
                            len(best_class_indices)), best_class_indices]
                        # print(best_class_probabilities)
                    except Exception as e:
                        # Error Logging : FOR DEVELOPMENT ONLY!!!
                        logging.error(traceback.format_exc())

                    # boxing face
                    cv2.rectangle(
                        frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)

                    # plot result idx under box
                    text_x = bb[i][0]
                    text_y = bb[i][3] + 20
                    # print('result: ', best_class_indices[0])
                    # print(best_class_indices)
                    # print(faceIds)
                    for faceId in faceIds:
                        # print(faceId)
                        if faceIds[best_class_indices[0]] == faceId and best_class_probabilities[0] > 0.7:
                            result_names = faceIds[best_class_indices[0]]
                            cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), thickness=1, lineType=2)

            else:
                print('[INFO] No detected face in the threshold vicinity!')

            # Show the frame
            cv2.imshow('Video', frame)

            # Quit if on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('[INFO] Stopping...')
                break

        video_capture.stop()
        cv2.destroyAllWindows()
