from flask import Blueprint, render_template

mod = Blueprint("home", __name__, template_folder="templates")

@mod.route("/")
def index():
    return render_template("home/index.html")
