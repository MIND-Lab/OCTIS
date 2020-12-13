from flask import Flask, render_template, request
import json
from multiprocessing import Process, Pool
import optopic.configuration.defaults as defaults
import optopic.dashboard.frameworkScanner as fs
from optopic.dashboard.queueManager import QueueManager
import webbrowser
import argparse

queueManager = QueueManager()

app = Flask(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, help="port", default=5000)
parser.add_argument("--host", type=str, help="host", default='localhost')


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/startExperiment', methods=['POST'])
def startExperiment():

    data = request.form.to_dict(flat=False)
    batch = data["batchId"][0]
    experimentId = data["expId"][0]
    expParams = {}
    expParams["path"] = data["path"][0]
    expParams["dataset"] = data["dataset"][0]
    expParams["model"] = {"name": data["model"][0]}
    expParams["optimization"] = {
        "iterations": typed(data["iterations"][0]),
        "model_runs": typed(data["runs"][0]),
        "surrogate_model": data["surrogateModel"][0],
        "n_random_starts": typed(data["n_random_starts"][0]),
        "acquisition_function": data["acquisitionFunction"][0],
        "search_spaces": {}
    }
    expParams["optimize_metrics"] = []
    expParams["track_metrics"] = []

    model_parameters_to_optimize = []

    for key, value in data.items():
        if "_check" in key:
            model_parameters_to_optimize.append(key.replace("_check", ''))

    for key, value in data.items():
        if "model." in key:
            if any(par in key for par in model_parameters_to_optimize):
                if("_min" in key):
                    name = key.replace("_min", '').replace("model.", '')
                    if name not in expParams["optimization"]["search_spaces"]:
                        expParams["optimization"]["search_spaces"][name] = {}
                    expParams["optimization"]["search_spaces"][name]["low"] = typed(
                        value[0])
                elif("_max" in key):
                    name = key.replace("_max", '').replace("model.", '')
                    if name not in expParams["optimization"]["search_spaces"]:
                        expParams["optimization"]["search_spaces"][name] = {}
                    expParams["optimization"]["search_spaces"][name]["high"] = typed(
                        value[0])
                elif("_check" not in key):
                    expParams["optimization"]["search_spaces"][key.replace(
                        "model.", '')] = request.form.getlist(key)
            else:
                if "name" in key:
                    expParams["model"][key.replace("model.", '')] = value[0]
                else:
                    if "parameters" not in expParams["model"]:
                        expParams["model"]["parameters"] = {}
                    expParams["model"]["parameters"][key.replace(
                        "model.", '')] = typed(value[0])

        if "metric." in key:
            optimize = True
            metric = {"name": key.replace("metric.", ''), "parameters": {}}
            for singleValue in value:

                for key, content in json.loads(singleValue).items():
                    if key != "metric" and key != "type":
                        metric["parameters"][key] = typed(content)
                    if key == "type" and content == "track":
                        optimize = False
                if optimize:
                    expParams["optimize_metrics"].append(metric)
                else:
                    expParams["track_metrics"].append(metric)

    print(expParams)

    queueManager.add_experiment(batch, experimentId, expParams)
    return CreateExperiments()


@app.route("/getBatchExperiments", methods=['POST'])
def getBatchExperiments():
    data = request.json['data']
    experiments = []
    for key in data:
        batchExperiments = queueManager.getBatchExperiments(key)
        for experiment in batchExperiments:
            newExp = experiment
            newExp["optimization_data"] = queueManager.getExperimentInfo(
                experiment)
            experiments.append(newExp)
    return json.dumps(experiments)


@ app.route('/CreateExperiments')
def CreateExperiments():
    models = defaults.model_hyperparameters
    datasets = fs.scanDatasets()
    metrics = defaults.metric_parameters
    optimization = defaults.optimization_parameters
    return render_template("CreateExperiments.html",
                           datasets=datasets,
                           models=models,
                           metrics=metrics,
                           optimization=optimization)


@ app.route('/VisualizeExperiments')
def VisualizeExperiments():
    batchNames = queueManager.getBatchNames()
    return render_template("VisualizeExperiments.html",
                           batchNames=batchNames)


@ app.route('/ManageExperiments')
def ManageExperiments():
    return render_template("ManageExperiments.html")


@ app.route('/SingleExperiment/<batch>/<id>')
def SingleExperiment(batch="", id=""):
    output = queueManager.getModel(batch, id, 0, 0)
    return render_template("SingleExperiment.html", batchName=batch, experimentName=id, output=output)


if __name__ == '__main__':
    args = parser.parse_args()

    url = 'http://' + str(args.host) + ':' + str(args.port)
    webbrowser.open_new(url)
    app.run(port=args.port)


def typed(value):
    try:
        typed = int(value)
        return typed
    except ValueError:
        try:
            typed = float(value)
            return typed
        except ValueError:
            return value
