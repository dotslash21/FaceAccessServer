import os
import sys
import shutil
from .. import app
from ..extensions import db
from ..models import User
from ..utils.train_svc import TrainSVC
from ..utils.align_img_db import AlignImgDB


train_svc = TrainSVC(
    app.config['ALIGNED_IMG_DB'],
    app.config['FACENET_PRETRAINED_MODEL_PATH'],
    app.config['SVC_CLASSIFIER_SAVE_PATH']
)
align_img_db = AlignImgDB(
    app.config['IMG_DB'],
    app.config['ALIGNED_IMG_DB'],
    app.config['MTCNN_MODEL_DIR']    
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_IMG_EXTENSIONS']

def delete_from_imgdb(foldername):
    directory1 = os.path.join(app.config['IMG_DB'], foldername)
    directory2 = os.path.join(app.config['ALIGNED_IMG_DB'], foldername)
    shutil.rmtree(directory1)
    shutil.rmtree(directory2)


def add_to_imgdb(foldername, files):
    for (index, file) in enumerate(files):
        if file and allowed_file(file.filename) and index < 10:
            directory = os.path.join(
                app.config['IMG_DB'], foldername)
            extension = file.filename.rsplit('.', 1)[1].lower()
            filename = foldername + "_{0}.{1}".format(index, extension)
            if not os.path.exists(directory):
                os.makedirs(directory)
            file.save(os.path.join(directory, filename))
            print("\'{0}\' has been saved at location \'{1}\'".format(
                filename, directory), file=sys.stdout)


def add_user_entry(first_name, last_name):
    try:
        user = User(first_name=first_name, last_name=last_name)
        db.session.add(user)
        db.session.commit()

        return 0
    except:
        raise Exception('[ERROR] Problem encountered while adding the user entry to database!')


def edit_user_entry(id, first_name, last_name):
    try:
        user = User.query.get_or_404(id)
        user.first_name = first_name
        user.last_name = last_name
        db.session.commit()

        return 0
    except:
        raise Exception('[ERROR] Problem encountered while editing the user entry in database!')


def delete_user_entry(user):
    try:
        db.session.delete(user)
        db.session.commit()

        return 0
    except:
        raise Exception('[ERROR] Problem encountered while deleting the user entry from database!')


def retrain_svc():
    align_img_db.perform_alignment()
    train_svc.train_svc()
    app.config['SVC_RELOAD'] = True