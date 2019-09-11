import os
import sys
import math
import pickle
import argparse
import numpy as np
import tensorflow as tf
import utils.facenet as facenet
import utils.detect_face as detect_face
from sklearn.svm import SVC

with tf.Graph().as_default():
    with tf.Session() as sess:
        datadir = './aligned_img_db/'
        dataset = facenet.get_dataset(datadir)
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))

        # Load the Facenet model and weights from file
        print('Loading feature extraction model...')
        modeldir = './pretrained_model/20170511-185253.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
        embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        embedding_size = embeddings.get_shape()[1]

        # Run a forward pass to calculate the image
        # vector embeddings.
        print('Calculating image features...')
        batch_size = 500  # 1000
        image_size = 160
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        embedding_arr = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = {images_placeholder: images,
                         phase_train_placeholder: False}
            embedding_arr[start_index:end_index, :] = sess.run(
                embeddings, feed_dict=feed_dict)

        # Setup the classifier for classifing new images
        # based on their embedding vector.
        classifier_filename = './my_classifier/SVC.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)

        # Train the classifier
        print('Training classifier...')
        model = SVC(kernel='linear', probability=True)
        model.fit(embedding_arr, labels)

        # Create the classname list
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # Save the classifier model to file
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)
