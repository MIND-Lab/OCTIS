from flask import Flask, render_template
from multiprocessing import Process, Pool
import frameworkScanner as fs

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/CreateExperiments')
def CreateExperiments():
    models = {
        "LDA": {"alpha": {
            "type": "real",
            "default_value": 0.1,
            "min_value": 1e-4,
            "max_value": 20},
            "beta": {
            "type": "real",
            "default_value": 0.1,
            "min_value": 1e-4,
            "max_value": 20}},
        "NMF": {"pizza": {
            "type": "categorical",
            "default_value": "margherita",
            "possible_values": ["kebab", "zola e noci", "4 formaggi"]
        }},
        "HDP": {"alpha": {
            "type": "integer",
            "default_value": 0,
            "min_value": 0,
            "max_value": 20},
            "beta": {
            "type": "real",
            "default_value": 0.1,
            "min_value": 1e-4,
            "max_value": 20},
            "pizza": {
            "type": "categorical",
            "default_value": "margherita",
            "possible_values": ["kebab", "zola e noci", "4 formaggi"]
        }}
    }
    datasets = fs.scanDatasets()
    return render_template("CreateExperiments.html", datasets=datasets, models=models)


@ app.route('/VisualizeExperiments')
def VisualizeExperiments():
    return render_template("VisualizeExperiments.html")


@ app.route('/ManageExperiments')
def ManageExperiments():
    return render_template("ManageExperiments.html")
