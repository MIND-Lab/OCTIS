from flask import Flask, render_template, request
from multiprocessing import Process, Pool
import optopic.configuration.defaults as defaults
import frameworkScanner as fs

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/startExperiment', methods=['POST'])
def startExperiment():
    data = request.form
    print(data)
    return CreateExperiments()


@app.route('/CreateExperiments')
def CreateExperiments():
    models = defaults.model_hyperparameters
    datasets = fs.scanDatasets()
    metrics = defaults.metric_parameters
    return render_template("CreateExperiments.html", datasets=datasets, models=models, metrics=metrics)


@ app.route('/VisualizeExperiments')
def VisualizeExperiments():
    return render_template("VisualizeExperiments.html")


@ app.route('/ManageExperiments')
def ManageExperiments():
    return render_template("ManageExperiments.html")
