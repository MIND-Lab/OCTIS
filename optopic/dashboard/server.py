from flask import Flask, render_template, request
from multiprocessing import Process, Pool
import optopic.configuration.defaults as defaults
import optopic.dashboard.frameworkScanner as fs
import webbrowser
import argparse


app = Flask(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, help="port", default=5000)
parser.add_argument("--host", type=str, help="host", default='localhost')


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
