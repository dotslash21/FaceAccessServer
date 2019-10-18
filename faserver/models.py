import datetime
from .extensions import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(40), nullable=False)
    last_name = db.Column(db.String(40), nullable=False)
    date_registered = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return "<USER-{0}>".format(self.id)