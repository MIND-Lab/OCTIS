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
        "iterations": data["iterations"][0],
        "model_runs": data["runs"][0],
        "surrogate_model": data["surrogateModel"][0],
        "acquisition_function": data["acquisitionFunction"][0]
    }
    expParams["optimize_metrics"] = []
    expParams["track_metrics"] = []

    for key, value in data.items():
        if "model." in key:
            expParams["model"][key.replace("model.", '')] = value[0]

        if "metric." in key:
            optimize = True
            metric = {"name": key.replace("metric.", '')}
            for key, content in json.loads(value[0]).items():
                if key != "metric" and key != "type":
                    metric[key] = content
                if key == "type" and content == "track":
                    optimize = False
            if optimize:
                expParams["optimize_metrics"].append(metric)
            else:
                expParams["track_metrics"].append(metric)
    
    # DA TERMINARE FORMATTAZIONE DATI: gestire spazi ricerca, tipo dato (string, numerico)
    print(batch)
    print()
    print(experimentId)
    print()
    print(expParams)
    print()

    queueManager.add_experiment(batch, experimentId, expParams)
    queueManager.save_state()
    return CreateExperiments()


@app.route('/CreateExperiments')
def CreateExperiments():
    models = defaults.model_hyperparameters
    datasets = fs.scanDatasets()
    metrics = defaults.metric_parameters
    optimization = {
        "surrogate_models": [{"name": "Gaussian proccess", "id": "GP"},
                             {"name": "Random forest", "id": "RF"}],
        "acquisition_functions": [{"name": "Upper confidence bound", "id": "UCB"},
                                  {"name": "Expected improvement", "id": "EI"}]
    }
    return render_template("CreateExperiments.html",
                           datasets=datasets,
                           models=models,
                           metrics=metrics,
                           optimization=optimization)


@ app.route('/VisualizeExperiments')
def VisualizeExperiments():
    return render_template("VisualizeExperiments.html")


@ app.route('/ManageExperiments')
def ManageExperiments():
    return render_template("ManageExperiments.html")


if __name__ == '__main__':
    args = parser.parse_args()

    url = 'http://' + str(args.host) + ':' + str(args.port)
    webbrowser.open_new(url)
    app.run(port=args.port)
