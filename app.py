import os
import sys
import shutil
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, url_for, request, redirect, flash

app = Flask(__name__)
app.config['IMG_DB'] = 'img_db'
app.config['ALIGNED_IMG_DB'] = 'aligned_img_db'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(40), nullable=False)
    last_name = db.Column(db.String(40), nullable=False)
    date_registered = db.Column(db.DateTime, default=datetime.utcnow)

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


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


if __name__ == '__main__':
    app.run(debug=True)
