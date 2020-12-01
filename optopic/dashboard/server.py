<<<<<<< HEAD
from flask import Flask, render_template, request
from multiprocessing import Process, Pool
import optopic.configuration.defaults as defaults
import frameworkScanner as fs
=======
from flask import Flask, render_template
import optopic.frameworkScanner as fs
>>>>>>> 51ef64b7149081839cb7a169e650736f4f3a1484

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
    metrics = {
        "metric_class_name": {
            "name": "metric 1",
            "texts": {
                "type": "String"
            },
            "topk": {
                "type": "Integer",
                "default_value": 10
            },
            "weight": {
                "type": "Real",
                "default_value": 3.1415
            },
            "aCategoricalThing": {
                "type": "Categorical",
                "default_value": "A",
                "possible_values": ["A", "B", "C"]
            }
        },
        "metric_class_name2": {
            "name": "metric 2",
            "texts": {
                "type": "String"
            },
            "topk": {
                "type": "Integer",
                "default_value": 23
            },
            "weight": {
                "type": "Real",
                "default_value": 2.7172
            },
            "aCategoricalThing": {
                "type": "Categorical",
                "default_value": "C",
                "possible_values": ["A", "B", "C"]
            }
        }
    }
    return render_template("CreateExperiments.html", datasets=datasets, models=models, metrics=metrics)


@ app.route('/VisualizeExperiments')
def VisualizeExperiments():
    return render_template("VisualizeExperiments.html")


@ app.route('/ManageExperiments')
def ManageExperiments():
    return render_template("ManageExperiments.html")
