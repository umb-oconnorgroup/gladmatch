import os

from flask import Flask


app = Flask(__name__)
# app.secret_key = os.environ.get('SECRET_KEY')
app.secret_key = "00328e042b1d9a75f86ff9ca547e6a1688b455f6b8a73784"
from app import routes
