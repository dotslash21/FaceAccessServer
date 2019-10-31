from .extensions import db
from flask import Flask

app = Flask(__name__)

# Load the configuration
app.config.from_pyfile("config.py")

from faserver.home.routes import mod as home_mod
from faserver.admin.routes import mod as admin_mod
from faserver.faceauth.routes import mod as faceauth_mod

app.register_blueprint(home_mod)
app.register_blueprint(admin_mod, url_prefix="/admin")
app.register_blueprint(faceauth_mod)


db.init_app(app)
