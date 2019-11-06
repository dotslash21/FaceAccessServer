import sys

BASE_PATH = sys.path[0]
IMG_DB = BASE_PATH + "/faserver/database/img_db"
ALIGNED_IMG_DB = BASE_PATH + "/faserver/database/aligned_img_db"
ALLOWED_IMG_EXTENSIONS = set(["png", "jpg", "jpeg", "gif"])
SQLALCHEMY_DATABASE_URI = "sqlite:///" + \
    BASE_PATH + "/faserver/database/database.db"
FACENET_PRETRAINED_MODEL_PATH = BASE_PATH + \
    "/faserver/utils/pretrained_model/20180402-114759/20180402-114759.pb"
MTCNN_MODEL_DIR = BASE_PATH + "/faserver/utils/npy"
SVC_CLASSIFIER_SAVE_PATH = BASE_PATH + "/faserver/utils/svc_classifier/SVC.pkl"
SVC_RELOAD = False
