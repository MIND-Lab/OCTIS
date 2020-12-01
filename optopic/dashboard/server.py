from flask import Flask, render_template
import optopic.frameworkScanner as fs

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/CreateExperiments')
def CreateExperiments():
    datasets = fs.scanDatasets()
    return render_template("CreateExperiments.html", datasets=datasets)


@app.route('/VisualizeExperiments')
def VisualizeExperiments():
    return render_template("VisualizeExperiments.html")


@app.route('/ManageExperiments')
def ManageExperiments():
    return render_template("ManageExperiments.html")
