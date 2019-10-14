from flask import Blueprint, render_template, Response
from .logic import generate


mod = Blueprint("faceid", __name__, template_folder="templates")

@mod.route("/view-feed")
def index():
    return render_template("faceid/view_feed.html")

@mod.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
