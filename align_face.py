import sys
import os
import argparse
import imageio
import numpy as np
from PIL import Image
import tensorflow as tf
import utils.facenet as facenet
import utils.detect_face as detect_face
import random
from time import sleep

# Set allow_pickle=True
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

output_dir_path = './aligned_img_db/'
output_dir = os.path.expanduser(output_dir_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

datadir = './img_db/'
dataset = facenet.get_dataset(datadir)

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './npy')

# Config variables
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
margin = 44
image_size = 182

# Add a random key to the filename to allow alignment using multiple processes
random_key = np.random.randint(0, high=99999)
bounding_boxes_filename = os.path.join(
    output_dir, 'bounding_boxes_%05d.txt' % random_key)

with open(bounding_boxes_filename, "w") as text_file:
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename + '.png')
            print("Source Image: %s" % image_path)
            if not os.path.exists(output_filename):
                try:
                    img = imageio.imread(image_path)
                    print('Read data dimension: ', img.ndim)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim < 2:
                        print('Error! Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                        print('to_rgb data dimension: ', img.ndim)
                    img = img[:, :, 0:3]
                    print('After data dimension: ', img.ndim)

                    bounding_boxes, _ = detect_face.detect_face(
                        img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print('Number of Detected Face(s): %d' % nrof_faces)
                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(img.shape)[0:2]
                        if nrof_faces > 1:
                            bounding_box_size = (
                                det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                            img_center = img_size / 2
                            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                            offset_dist_squared = np.sum(
                                np.power(offsets, 2.0), 0)
                            # some extra weight on the centering
                            index = np.argmax(
                                bounding_box_size - offset_dist_squared * 2.0)
                            det = det[index, :]
                        det = np.squeeze(det)
                        bb_temp = np.zeros(4, dtype=np.int32)

                        bb_temp[0] = det[0]
                        bb_temp[1] = det[1]
                        bb_temp[2] = det[2]
                        bb_temp[3] = det[3]

                        try:
                            cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                            # scaled_temp = misc.imresize(
                            #     cropped_temp, (image_size, image_size), interp='bilinear')
                            scaled_temp = np.array(Image.fromarray(cropped_temp).resize(
                                (image_size, image_size), resample=Image.BILINEAR))

                            nrof_successfully_aligned += 1
                            imageio.imsave(output_filename, scaled_temp)
                            text_file.write('%s %d %d %d %d\n' % (
                                output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))
                        except Exception as e:
                            os.remove(image_path)
                    else:
                        print('Error! Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))

print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
