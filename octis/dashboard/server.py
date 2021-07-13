import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
octispath = Path(os.path.dirname(__file__)).parent.absolute().parent.absolute()
sys.path.append(str(octispath))

import argparse
import webbrowser
import frameworkScanner as fs
import octis.configuration.defaults as defaults
from multiprocessing import Process, Pool
import json
from flask import Flask, render_template, request, send_file
import tkinter as tk
import pandas as pd
import numpy as np
from tkinter import filedialog


app = Flask(__name__)
queueManager = ""


@app.route('/downloadSingleExp',
           methods=['GET'])
def downloadSingleExp():
    """
    Download the report of the given experiment

    :erturn: file with the report
    :rtype: File
    """
    experimentId = request.args.get("experimentId")
    batchId = request.args.get("batchId")

    paused = False
    if (experimentId + batchId) == queueManager.running:
        paused = True
        queueManager.pause()

    expPath = ""
    createdPath = os.path.join(
        queueManager.getExperiment(batchId, experimentId)["path"],
        experimentId,
        experimentId)
    jsonReport = {}
    if os.path.isfile(createdPath+".json"):
        expPath = createdPath

    with open(expPath+".json") as p:
        jsonReport = json.load(p)

    info = queueManager.getExperimentInfo(batchId, experimentId)

    n_row = info["current_iteration"]
    n_extra_metrics = len(jsonReport["extra_metric_names"])

    df = pd.DataFrame()
    df['dataset'] = [jsonReport["dataset_name"]] * n_row
    df['surrogate model'] = [jsonReport["surrogate_model"]] * n_row
    df['acquisition function'] = [jsonReport["acq_func"]] * n_row
    df['num_iteration'] = [i for i in range(n_row)]
    df['time'] = [jsonReport['time_eval'][i] for i in range(n_row)]
    df['Median(model_runs)'] = [np.median(
        jsonReport['dict_model_runs'][jsonReport['metric_name']]['iteration_' + str(i)]) for i in range(n_row)]
    df['Mean(model_runs)'] = [np.mean(
        jsonReport['dict_model_runs'][jsonReport['metric_name']]['iteration_' + str(i)]) for i in range(n_row)]
    df['Standard_Deviation(model_runs)'] = [np.std(
        jsonReport['dict_model_runs'][jsonReport['metric_name']]['iteration_' + str(i)]) for i in range(n_row)]

    for hyperparameter in list(jsonReport["x_iters"]):
        df[hyperparameter] = jsonReport["x_iters"][hyperparameter][0:n_row]

    for metric, i in zip(jsonReport["extra_metric_names"], range(n_extra_metrics)):
        df[metric + '(median, not optimized)'] = [np.median(
            jsonReport["dict_model_runs"][metric]['iteration_' + str(i)]) for i in range(n_row)]

        df[metric + '(Mean, not optimized)'] = [np.mean(
            jsonReport["dict_model_runs"][metric]['iteration_' + str(i)]) for i in range(n_row)]

        df[metric + '(Standard_Deviation, not optimized)'] = [np.std(
            jsonReport["dict_model_runs"][metric]['iteration_' + str(i)]) for i in range(n_row)]

    name_file = expPath + ".csv"

    df.to_csv(name_file, index=False, na_rep='Unkown')

    if paused:
        queueManager.start()

    return send_file(expPath+".csv",
                     mimetype="text/csv",
                     attachment_filename="report.csv",
                     as_attachment=True)


@ app.route("/selectPath", methods=['POST'])
def selectPath():
    """
    Select a path from the server and return it to the page

    :return: path
    :rtype: Dict
    """
    window = tk.Tk()
    path = filedialog.askdirectory()
    window.destroy()
    return {"path": path}


@ app.route("/serverClosed")
def serverClosed():
    """
    Reroute to the serverClosed page before server shutdown

    :return: template
    :rtype: render template
    """
    return render_template("serverClosed.html")


@ app.route("/shutdown")
def shutdown():
    """
    Save the state of the QueueManager and perform server shutdown

    :return: Ack signal
    :rtype: Dict
    """
    queueManager.stop()
    shutdown_server()
    return {"DONE": "YES"}


@ app.route('/')
def home():
    """
    Return the octis landing page

    :return: template
    :rtype: render template
    """
    return render_template("index.html")


@ app.route('/startExperiment', methods=['POST'])
def startExperiment():
    """
    Add a new experiment to the queue

    :return: template
    :rtype: render template
    """
    data = request.form.to_dict(flat=False)
    batch = data["batchId"][0]
    experimentId = data["expId"][0]
    if queueManager.getExperiment(batch, experimentId):
        return VisualizeExperiments()

    expParams = dict()
    expParams["partitioning"] = ("partitioning" in data)
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
                if "_xminx" in key:
                    name = key.replace("_xminx", '').replace("model.", '')
                    if name not in expParams["optimization"]["search_spaces"]:
                        expParams["optimization"]["search_spaces"][name] = {}
                    expParams["optimization"]["search_spaces"][name]["low"] = typed(
                        value[0])
                elif "_xmaxx" in key:
                    name = key.replace("_xmaxx", '').replace("model.", '')
                    if name not in expParams["optimization"]["search_spaces"]:
                        expParams["optimization"]["search_spaces"][name] = {}
                    expParams["optimization"]["search_spaces"][name]["high"] = typed(
                        value[0])
                elif "_check" not in key:
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



    if expParams["optimize_metrics"][0]["name"]=="F1Score" and not expParams["partitioning"]:
        return VisualizeExperiments()
    for trackedMetric in expParams["track_metrics"]:
        if trackedMetric["name"] == "F1Score" and not expParams["partitioning"]:
            return VisualizeExperiments()

    print(expParams)

    queueManager.add_experiment(batch, experimentId, expParams)
    return CreateExperiments()


@ app.route("/getBatchExperiments", methods=['POST'])
def getBatchExperiments():
    """
    return the information related to the experiments of a batch

    :return: informations of the experiment
    :rtype: Dict
    """
    data = request.json['data']
    experiments = []
    for key in data:
        batch_experiments = queueManager.getBatchExperiments(key)
        for experiment in batch_experiments:
            new_exp = experiment
            new_exp["optimization_data"] = queueManager.getExperimentInfo(
                experiment["batchId"],
                experiment["experimentId"])
            experiments.append(new_exp)
    return json.dumps(experiments)


@ app.route('/CreateExperiments')
def CreateExperiments():
    """
    Serve the experiment creation page

    :return: template
    :rtype: render template
    """
    models = defaults.model_hyperparameters
    models_descriptions = defaults.model_descriptions
    datasets = fs.scanDatasets()
    metrics = defaults.metric_parameters
    optimization = defaults.optimization_parameters
    return render_template("CreateExperiments.html",  datasets=datasets, models=models, metrics=metrics,
                           optimization=optimization, models_descriptions=models_descriptions)


@ app.route('/VisualizeExperiments')
def VisualizeExperiments():
    """
    Serve the experiments visualization page

    :return: template
    :rtype: render template
    """
    batch_names = queueManager.getBatchNames()
    return render_template("VisualizeExperiments.html",
                           batchNames=batch_names)


@ app.route('/ManageExperiments')
def ManageExperiments():
    """
    Serve the ManageExperiments page

    :return: template
    :rtype: render template
    """
    exp_list = queueManager.getToRun()
    for exp in exp_list:
        exp_info = queueManager.getExperimentInfo(
            exp_list[exp]["batchId"], exp_list[exp]["experimentId"])
        if exp_info is not None:
            exp_list[exp].update(exp_info)
    order = queueManager.getOrder()
    running = queueManager.getRunning()
    return render_template("ManageExperiments.html", order=order, experiments=exp_list, running=running)


@ app.route("/pauseExp", methods=["POST"])
def pauseExp():
    """
    Pause the current experiment

    :return: ack signal
    :rtype: Dict
    """
    queueManager.pause()
    return {"DONE": "YES"}


@ app.route("/startExp", methods=["POST"])
def startExp():
    """
    Start the next experiment in the queue

    :return: ack signal
    :rtype: Dict
    """
    print(queueManager.getRunning())
    if queueManager.getRunning() == None:
        queueManager.next()
    return {"DONE": "YES"}


@ app.route("/deleteExp", methods=["POST"])
def deleteExp():
    """
    Delete the selected experiment from the queue

    :return: ack signal
    :rtype: Dict
    """
    data = request.json['data']
    print(queueManager.getRunning())
    if queueManager.getRunning() is not None and queueManager.getRunning() == data:
        queueManager.pause()
        queueManager.deleteFromOrder(data)
    else:
        queueManager.deleteFromOrder(data)
    return {"DONE": "YES"}


@ app.route("/updateOrder", methods=["POST"])
def updateOrder():
    """
    Update the order of the experiments in the queue

    :return: ack signal
    :rtype: Dict
    """
    data = request.json['data']
    queueManager.editOrder(data)
    return {"DONE": "YES"}


@ app.route("/getDocPreview", methods=["POST"])
def getDocPreview():
    """
    Returns the first 40 words of the selected document

    :return: first 40 words of the document
    :rtype: Dict
    """
    data = request.json['data']
    return json.dumps({"doc": fs.getDocPreview(data["dataset"], int(data["document"]))})


@ app.route('/SingleExperiment/<batch>/<exp_id>')
def SingleExperiment(batch="", exp_id=""):
    """
    Serve the single experiment page

    :return: template
    :rtype: render template
    """
    models = defaults.model_hyperparameters
    output = queueManager.getModel(batch, exp_id, 0, 0)
    global_info = queueManager.getExperimentInfo(batch, exp_id)
    iter_info = queueManager.getExperimentIterationInfo(batch, exp_id, 0)
    exp_info = queueManager.getExperiment(batch, exp_id)
    exp_ids = queueManager.getAllExpIds()
    vocabulary_path = os.path.join(exp_info["path"],
                                   exp_info["experimentId"],
                                   "models",
                                   "vocabulary.json")
    vocabulary = fs.getVocabulary(vocabulary_path)

    return render_template("SingleExperiment.html", batchName=batch, experimentName=exp_id,
                           output=output, globalInfo=global_info, iterationInfo=iter_info,
                           expInfo=exp_info, expIds=exp_ids, datasetMetadata=fs.getDatasetMetadata(
                               exp_info["dataset"]), vocabulary=vocabulary, models=models)


@ app.route("/getIterationData", methods=["POST"])
def getIterationData():
    """
    Return data of a single iteration and model run of an experiment

    :return: data of a single iteration and model run of an experiment
    :rtype: Dict
    """
    data = request.json['data']
    output = queueManager.getModel(data["batchId"], data["experimentId"],
                                   int(data["iteration"]), data["model_run"])
    iter_info = queueManager.getExperimentIterationInfo(data["batchId"], data["experimentId"],
                                                        int(data["iteration"]))
    return {"iterInfo": iter_info, "output": output}


def typed(value):
    """
    Handles typing of data

    :param value: value to cast
    :type value: *

    :raises ValueError: cannot cast data

    :return: data with the right type
    :rtype: *
    """
    try:
        t = int(value)
        return t
    except ValueError:
        try:
            t = float(value)
            return t
        except ValueError:
            return value


def shutdown_server():
    """
    Perform server shutdown

    :raise RuntimeError: wrong server environment used
    """
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


if __name__ == '__main__':
    """
    Initialize the server
    """
    from queueManager import QueueManager

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help="port", default=5000)
    parser.add_argument("--host", type=str, help="host", default='localhost')
    parser.add_argument("--dashboardState", type=str, help="dashboardState", default="")

    args = parser.parse_args()

    dashboardState = None
    if args.dashboardState != "":
        dashboardState = args.dashboardState
    else:
        dashboardState = os.path.join(os.getcwd(),"queueManagerState.json")

    queueManager = QueueManager(dashboardState)

    url = 'http://' + str(args.host) + ':' + str(args.port)
    webbrowser.open_new(url)
    app.run(port=args.port)
