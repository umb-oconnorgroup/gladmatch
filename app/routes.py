import os
import tempfile
import uuid

from flask import jsonify, render_template, request, send_file

from app import app
from app.tasks import match as celery_match, app as celery_app


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/match", methods=["POST"])
def match():
    file_bytes = request.data
    file_id = str(uuid.uuid1())
    # will need to change to more stable temporary directory
    with open(os.path.join(tempfile.gettempdir(), file_id), "wb") as f:
        f.write(file_bytes)
    task = celery_match.delay(file_id)
    return jsonify({"task_id": task.id}), 202

@app.route("/tasks/<task_id>", methods=["GET"])
def get_status(task_id):
    task_result = celery_app.AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": task_result.status
    }
    return jsonify(result), 200

@app.route("/retrieve/<task_id>", methods=["GET"])
def retrieve(task_id):
    try:
        task_result = celery_app.AsyncResult(task_id)
    except:
        return render_template("404.html", title = "404"), 404
    if task_result.status != "SUCCESS" or task_result.result + ".npz" not in os.listdir(tempfile.gettempdir()):
        return render_template("404.html", title = "404"), 404
    else:
        return send_file(os.path.join(tempfile.gettempdir(), task_result.result + ".npz"), attachment_filename="allele_frequencies.npz", as_attachment=True)