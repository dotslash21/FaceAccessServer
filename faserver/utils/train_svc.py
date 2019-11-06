import math
import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from ..utils import detect_face, facenet


class TrainSVC:
    def __init__(self, datadir, modeldir, classifier_filename):
        # Config variables
        self.batch_size = 500  # 1000
        self.image_size = 160
        self.datadir = datadir
        self.modeldir = modeldir
        self.classifier_filename = classifier_filename

    def load_dataset(self):
        self.dataset = facenet.get_dataset(self.datadir)
        self.paths, self.labels = facenet.get_image_paths_and_labels(
            self.dataset)
        self.nrof_images = len(self.paths)
        self.nrof_batches_per_epoch = int(
            math.ceil(1.0 * self.nrof_images / self.batch_size))
        print('Number of classes: %d' % len(self.dataset))
        print('Number of images: %d' % len(self.paths))

    def load_embeddings_arr(self):
        self.embedding_arr = np.zeros((self.nrof_images, self.embedding_size))
        for i in range(self.nrof_batches_per_epoch):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, self.nrof_images)
            paths_batch = self.paths[start_index:end_index]
            images = facenet.load_data(
                paths_batch, False, False, self.image_size)
            feed_dict = {self.images_placeholder: images,
                         self.phase_train_placeholder: False}
            self.embedding_arr[start_index:end_index, :] = self.sess.run(
                self.embeddings, feed_dict=feed_dict)

    def train_svc(self):
        with tf.Graph().as_default():
            with tf.Session() as self.sess:
                # Load Dataset
                self.load_dataset()

                # Load the Facenet model and weights from file
                print('Loading feature extraction model...')
                facenet.load_model(self.modeldir)

                self.images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
                self.embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
                self.phase_train_placeholder = tf.get_default_graph(
                ).get_tensor_by_name('phase_train:0')
                self.embedding_size = self.embeddings.get_shape()[1]

                # Run a forward pass to calculate the image
                # vector embeddings.
                print('Calculating image features...')
                self.load_embeddings_arr()

                # Setup the classifier for classifing new images
                # based on their embedding vector.
                self.classifier_filename_exp = os.path.expanduser(
                    self.classifier_filename)

                # Train the classifier
                print('Training classifier...')
                model = OneVsRestClassifier(SVC(kernel='linear', probability=True))
                model.fit(self.embedding_arr, self.labels)

                # Create the classname list
                class_names = [
                    cls.name.replace(
                        '_', ' ') for cls in self.dataset]

                # Save the classifier model to file
                with open(self.classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print(
                    'Saved classifier model to file "%s"' %
                    self.classifier_filename_exp)
