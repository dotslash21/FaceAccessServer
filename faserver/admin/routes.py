from ..models import User
from .logic import allowed_file, delete_from_imgdb, add_to_imgdb, add_user_entry, edit_user_entry, delete_user_entry, retrain_svc
from flask import Blueprint, render_template, request, redirect


mod = Blueprint("admin", __name__, template_folder="templates")


@mod.route("/", methods=['GET'])
def index():
    users_count = User.query.count()
    return render_template('admin/dashboard.html', users_count=users_count)


@mod.route('/add-user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        uploaded_files = request.files.getlist("images")

        # try:
        add_user_entry(first_name, last_name)

        foldername = "{0}_{1}".format(first_name, last_name)
        add_to_imgdb(foldername, uploaded_files)

        retrain_svc()

        return redirect('/admin/add-user')
        # except:
        #     return '[ERROR] There was a problem adding that user!'
    else:
        return render_template('admin/add_user.html')


@mod.route('/edit-user', methods=['GET', 'POST'])
def edit_user():
    if request.method == 'POST':
        id = request.form['user_id']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        old_first_name = request.form['old_first_name']
        old_last_name = request.form['old_last_name']
        uploaded_files = request.files.getlist("images")
        
        # try:
        edit_user_entry(id, first_name, last_name)

        old_folder_name = "{0}_{1}".format(old_first_name, old_last_name)
        delete_from_imgdb(old_folder_name)

        new_folder_name = "{0}_{1}".format(first_name, last_name)
        add_to_imgdb(new_folder_name, uploaded_files)

        retrain_svc()

        return redirect('/admin/edit-user')
        # except:
        #     return '[ERROR] There was a problem editing that user!'
    else:
        users = User.query.order_by(User.date_registered).all()
        return render_template('admin/edit_user.html', users=users)


@mod.route('/delete-user', methods=['GET', 'POST'])
def delete_user():
    if request.method == 'POST':
        id = request.form['user_id']
        user = User.query.get_or_404(id)

        # try:
        # Delete user entry from database
        delete_user_entry(user)

        foldername = "{0}_{1}".format(user.first_name, user.last_name)
        delete_from_imgdb(foldername)

        retrain_svc()

        return redirect('/admin/delete-user')
        # except:
        #     return '[ERROR] There was a problem deleting that user!'
    else:
        users = User.query.order_by(User.date_registered).all()
        return render_template('admin/delete_user.html', users=users)