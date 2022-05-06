import os
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
    task_id = str(uuid.uuid1())
    task_data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "task_data")
    task_data_dir = os.path.realpath(task_data_dir)
    os.makedirs(os.path.join(task_data_dir, task_id))
    query_path = os.path.join(task_data_dir, task_id, "query.npy")
    result_path = os.path.join(task_data_dir, task_id, "result.vcf")
    with open(query_path, "wb") as f:
        f.write(file_bytes)
    task = celery_match.apply_async((query_path, result_path), task_id=task_id)
    return jsonify({"task_id": task.id}), 202

@app.route("/tasks/<task_id>", methods=["GET"])
def get_status(task_id):
    task_result = celery_app.AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": task_result.status
    }
    if task_result.status == "SUCCESS":
        result["quality_score"] = task_result.result
    return jsonify(result), 200

@app.route("/retrieve/<task_id>", methods=["GET"])
def retrieve(task_id):
    task_data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "task_data")
    try:
        task_result = celery_app.AsyncResult(task_id)
    except:
        return render_template("404.html", title = "404"), 404
    if task_result.status != "SUCCESS" or "result.vcf" not in os.listdir(os.path.join(task_data_dir, task_id)):
        return render_template("404.html", title = "404"), 404
    else:
        return send_file(os.path.join(task_data_dir, task_id, "result.vcf"), attachment_filename="result.vcf", as_attachment=True)