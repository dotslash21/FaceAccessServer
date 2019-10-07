import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
import utils.facenet as facenet
import utils.detect_face as detect_face
from PIL import Image

# Set allow_pickle=True
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


class FaceIdentifier:
    def __init__(self):
        print('[INFO] Initializing networks and loading parameters...')

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options, log_device_placement=False))
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(
                    self.sess, './npy')

                # Load dev variables
                print('[DEBUG] Loading development environment variables... ', end='')
                self.modeldir = './pretrained_model/20170511-185253.pb'
                self.svc_classifier_filename = './my_classifier/SVC.pkl'
                print('Done!')

                # Config variables
                print('[INFO] Initializing config parameters... ', end='')
                self.minsize = 150  # minimum size of face
                self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                self.factor = 0.709  # scale factor
                self.margin = 44
                self.frame_interval = 3
                self.image_size = 182
                self.batch_size = 1000
                self.input_image_size = 160
                print('Done!')

                # get the face IDs form the db
                print('[INFO] Loading face Ids from Database...', end='')
                self.faceIds = sorted(os.listdir('./aligned_img_db/'))
                for faceId in self.faceIds:
                    if(faceId[:15] == "bounding_boxes_"):
                        self.faceIds.remove(faceId)
                print('Done!')

                # Load the facenet model and weights from file
                print('[INFO] Loading feature extraction model...')
                facenet.load_model(self.modeldir)
                print('Done!')

                # Store references to layers in their
                # respective variables.
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph(
                ).get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]

                # Load the trained SVM Classifier from file
                classifier_filename_exp = os.path.expanduser(
                    self.svc_classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (self.model, self.class_names) = pickle.load(infile)
                    print('[INFO] Loaded SVC classifier file -> %s' %
                          classifier_filename_exp)

    # Function for performing face identification
    # Return an object containing:-
    #     -> Id
    #     -> Bounding Box Co-ordinates
    def identify(self, frame):

        # Force frame to have RGB channels
        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        frame = frame[:, :, 0:3]

        # Get the bounding boxes
        bounding_boxes, _ = detect_face.detect_face(
            frame, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)

        # Get the number of faces detected
        nrof_faces = bounding_boxes.shape[0]

        if nrof_faces > 1:
            print('[AUTH_ERROR] Multiple equidistant faces detected!')
            print('[AUTH_ERROR] Please maintain one-by-one queue!')
            return 1
        elif nrof_faces == 1:
            bb = np.zeros((nrof_faces, 4), dtype=np.int32)
            bb[0][0] = bounding_boxes[0][0]
            bb[0][1] = bounding_boxes[0][1]
            bb[0][2] = bounding_boxes[0][2]
            bb[0][3] = bounding_boxes[0][3]

            # Get the frame size
            img_size = np.asarray(frame.shape)[0:2]

            # For storing cropped, scaled and scaled+reshaped image
            cropped = None
            scaled = None
            scaled_reshape = None

            # Create Embedding array
            emb_array = np.zeros((1, self.embedding_size))

            # Bounding box out of frame size range exception
            if bb[0][0] <= 0 or bb[0][1] <= 0 or bb[0][2] >= len(frame[0]) or bb[0][3] >= len(frame[1]):
                print('[ERROR] Bounding Box out of frame size range!')
                return 2

            try:
                cropped = frame[bb[0][1]:bb[0][3], bb[0][0]:bb[0][2], :]
                cropped = facenet.flip(cropped, False)

                scaled = np.array(Image.fromarray(cropped).resize(
                    (self.image_size, self.image_size), resample=Image.BILINEAR))
                scaled = cv2.resize(
                    scaled, (self.input_image_size, self.input_image_size), interpolation=cv2.INTER_CUBIC)
                scaled = facenet.prewhiten(scaled)

                scaled_reshape = scaled.reshape(
                    -1, self.input_image_size, self.input_image_size, 3)

                feed_dict = {self.images_placeholder: scaled_reshape,
                             self.phase_train_placeholder: False}

                emb_array[0, :] = self.sess.run(
                    self.embeddings, feed_dict=feed_dict)

                predictions = self.model.predict_proba(emb_array)
                # print('predictions:', predictions)

                best_class_indices = np.argmax(predictions, axis=1)
                # print('best_class_indices:', best_class_indices)

                best_class_probabilities = predictions[np.arange(
                    len(best_class_indices)), best_class_indices]
                # print('best_class_probabilities:', best_class_probabilities)

            except Exception as e:
                print('[ERROR]', e)
                return 3

            if best_class_probabilities[0] > 0.7:
                return (self.faceIds[best_class_indices[0]], bb, best_class_probabilities[0])
            else:
                print('[AUTH_ERROR] ACCESS DENIED!')
                return 4

        else:
            print('[INFO] No detected face in the threshold vicinity!')
            return 5
