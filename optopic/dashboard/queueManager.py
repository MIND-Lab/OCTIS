import os
import time
import json
from pathlib import Path
from collections import namedtuple
from optopic.dashboard.experimentManager import startExperiment
import optopic.dashboard.experimentManager as expManager
import multiprocessing as mp
from subprocess import Popen


class QueueManager:
    running = mp.Manager().list()
    running.append(None)
    toRun = mp.Manager().dict()
    order = mp.Manager().list()
    completed = mp.Manager().dict()
    process = None
    busy = mp.Manager().list()
    busy.append(False)
    idle = None

    def __init__(self):
        """
        Initialize the queue manager.
        Loads old queues
        """
        self.load_state()
        self.idle = mp.Process(target=self._run)
        self.idle.start()

    def save_state(self):
        """
        Saves the state of the queue
        """
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        path = os.path.join(path, "queueManagerState.json")
        with open(path, "w") as fp:
            json.dump({"running": self.running[0],
                       "toRun": dict(self.toRun),
                       "order": list(self.order),
                       "completed": dict(self.completed)},
                      fp)

    def load_state(self):
        """
        Loads the state of the queue
        """
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        path = os.path.join(path, "queueManagerState.json")
        with open(path, "r") as fp:
            data = json.load(fp)
            self.running[0] = data["running"]
            self.toRun.update(data["toRun"])
            self.order.extend(data["order"])
            self.completed.update(data["completed"])

    def next(self):
        """
        If there is no running experiments, choose te next one to run

        Returns
        -------
        output : a tuple containing id of the batch and id of
                 the next experiment to run
        """
        if self.running[0] is None:
            self.running[0] = self.order.pop(0)
            self.start()
        return self.running[0]

    def add_experiment(self, batch, id, parameters):
        """
        Adds a new experiment to the queue

        Parameters
        ----------
        batch : id of the batch
        id : id of the experiment
        parameters : dictionary with the parameters of the experiment

        Returns
        -------
        True if the experiment was added to the queue, False otherwise
        """
        toAdd = batch+id
        parameters["batchId"] = batch
        parameters["experimentId"] = id
        if toAdd not in self.completed and toAdd not in self.toRun:
            self.toRun[toAdd] = parameters
            self.order.append(toAdd)
            return True
        return False

    def _run(self):
        """
        Put the current experiment in the finished queue

        Returns
        -------
        output : a tuple containing id of the batch and id of the
                 completed experiment
        """
        while(True):
            time.sleep(7)
            if not self.busy[0]:
                if self.running[0] is not None:
                    finished = self.running[0]
                    self.completed[finished] = self.toRun[finished]
                    del self.toRun[finished]
                    self.running[0] = None
                    self.save_state()
                if len(self.order) > 0 and self.running[0] is None:
                    self.running[0] = self.order.pop(0)
                    self.start()

    def pause(self):
        """
        pause the running experiment

        Returns
        -------
        output : a tuple containing id of the batch and id of the
                 paused experiment
        """
        if self.busy[0]:
            paused = self.running[0]
            self.process.terminate()
            self.order.insert(0, paused)
            self.running[0] = None
            return paused
        return False

    def getBatchNames(self):
        batchNames = []
        for key, value in self.completed.items():
            if value["batchId"] not in batchNames:
                batchNames.append(value["batchId"])
        for key, value in self.toRun.items():
            if value["batchId"] not in batchNames:
                batchNames.append(value["batchId"])
        return batchNames

    def getBatchExperiments(self, batchName):
        experiments = []
        for key, value in self.completed.items():
            if value["batchId"] == batchName:
                experiments.append(value)
        for key, value in self.toRun.items():
            if value["batchId"] == batchName:
                experiments.append(value)
        return experiments

    def getExperimentInfo(self, experiment):
        path = str(os.path.join(
            experiment["path"], experiment["experimentId"], experiment["experimentId"]+".json"))
        return expManager.retrieveBoResults(path)

    def start(self):
        if not self.busy[0]:
            self.busy[0] = True
            self.process = mp.Process(target=self._execute_and_update)
            self.process.start()

    def stop(self):
        self.idle.terminate()
        self.pause()
        self.save_state()

    def _execute_and_update(self):
        startExperiment(self.toRun[self.running[0]])
        self.busy[0] = False

    def getModel(self, batch, experiment_id, iteration, model_run):
        experiment = self.completed[batch + experiment_id]
        path = str(os.path.join(
            experiment["path"], experiment["experimentId"]))
        return expManager.getModelInfo(path, iteration, model_run)
