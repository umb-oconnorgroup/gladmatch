from email.mime.text import MIMEText
import json
import os
import smtplib
import ssl
import subprocess

from celery import Celery
import yaml


with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yaml"), "r") as config_file:
    config = yaml.safe_load(config_file)


app = Celery('tasks', backend="redis://localhost", broker="pyamqp://")

@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(60.0, scan_tasks.s(), expires=30)

@app.task
def scan_tasks():
    task_data_dir = config["tasks"]["data_dir"]
    for task_id in os.listdir(task_data_dir):
        task_result = app.AsyncResult(task_id)
        if task_result.state == "PENDING":
            match.apply_async((task_id,), task_id=task_id)
            continue
        elif task_result.state == "SUCCESS":
            if app.AsyncResult("notif" + task_id[:-5]).state == "PENDING":
                job_id = task_result.result
                process = subprocess.run(["qstat", "-j", job_id], capture_output=True)
                if process.stderr.decode().startswith("Following jobs do not exist"):
                    notify.apply_async((task_id,), task_id="notif" + task_id[:-5])
                continue

@app.task
def notify(task_id):
    port = 465
    smtp_server = "smtp.gmail.com"
    sender_email = config["email"]["address"]
    password = config["email"]["password"]
    task_data_dir = config["tasks"]["data_dir"]
    with open(os.path.join(task_data_dir, task_id, "params.json"), "r") as params_file:
        receiver_email = json.load(params_file)["email"]

    if "match.vcf" in os.listdir(os.path.join(task_data_dir, task_id)):
        subject = "Subject: GLADdb Matching Complete - {}".format(task_id)
        body = """The matching task you submitted has completed successfully.
        You can download the results using the following link:
        https://gladdb.igs.umaryland.edu/tasks/{}""".format(task_id)
    else:
        subject = "Subject: GLADdb Matching Failed - {}".format(task_id)
        body = """The matching task you submitted has failed."""
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())


@app.task
def match(task_id):
    bash_script_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "match.sh")
    python_script_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "match.py")
    config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yaml")
    task_dir = os.path.join(config["tasks"]["data_dir"], task_id)
    query_path = os.path.join(task_dir, "query.npz")
    result_path = os.path.join(task_dir, "match.vcf")
    with open(os.path.join(task_dir, "params.json"), "r") as params_file:
        params = json.load(params_file)
        if params["n"] == "":
            n = "-1"
        else:
            try:
                n = int(params["n"])
                n = str(n)
            except:
                n = "-1"
        exclude_phs = params["exclude-phs"]
        if "self-described" in params:
            self_described = "--self-described"
        else:
            self_described = ""

    subprocess.run(["cp", python_script_path, task_dir])
    subprocess.run(["cp", config_path, task_dir])
    process = subprocess.run(["qsub", "-wd", task_dir, bash_script_path, "match.py", "config.yaml", query_path, result_path, n, exclude_phs, self_described], capture_output=True)
    if process.stdout.decode().startswith("Your job"):
        # return job id
        return process.stdout.decode().split()[2]
    else:
        raise ValueError("Unexpected qsub output.\nstdout: {}\nstderr".format(process.stdout.decode(), process.stderr.decode()))