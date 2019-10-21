from flask import Flask, redirect

app = Flask(__name__)

#Load the configuration
app.config.from_pyfile("config.py")


from faserver.home.routes import mod as home_mod
from faserver.rt_faceauth.routes import mod as rt_faceauth_mod
from faserver.admin.routes import mod as admin_mod

app.register_blueprint(home_mod)
app.register_blueprint(rt_faceauth_mod)
app.register_blueprint(admin_mod, url_prefix="/admin")


from .extensions import db

db.init_app(app)