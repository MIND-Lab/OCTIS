import os
import json
from pathlib import Path
from collections import namedtuple
from optopic.dashboard.experimentManager import startExperiment
import multiprocessing as mp


class QueueManager:
    running = None
    toRun = {}
    order = []
    completed = {}
    process = mp.Pool(processes=1)

    def __init__(self):
        """
        Initialize the queue manager.
        Loads old queues
        """
        self.load_state()

    def save_state(self):
        """
        Saves the state of the queue
        """
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        path = os.path.join(path, "queueManagerState.json")
        with open(path, "w") as fp:
            json.dump({"running": self.running,
                       "toRun": self.toRun,
                       "order": self.order,
                       "completed": self.completed},
                      fp)

    def load_state(self):
        """
        Loads the state of the queue
        """
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        path = os.path.join(path, "queueManagerState.json")
        with open(path, "r") as fp:
            data = json.load(fp)
            self.running = data["running"]
            self.toRun = data["toRun"]
            self.order = data["order"]
            self.completed = data["completed"]

    def next(self):
        """
        If there is no running experiments, choose te next one to run

        Returns
        -------
        output : a tuple containing id of the batch and id of
                 the next experiment to run
        """
        if self.running == None:
            self.running = self.order.pop(0)
            self.start()
        return self.running

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

    def finished(self):
        """
        Put the current experiment in the finished queue

        Returns
        -------
        output : a tuple containing id of the batch and id of the
                 completed experiment
        """
        finished = self.running
        self.completed[finished] = self.toRun[finished]
        del self.toRun[finished]
        self.running = None
        return finished

    def pause(self):
        """
        pause the running experiment

        Returns
        -------
        output : a tuple containing id of the batch and id of the
                 paused experiment
        """
        if self.process != None:
            paused = self.running
            self.process.terminate()
            self.order.insert(0, paused)
            self.running = None
            return paused
        return False

    def start(self):
        if self.process == None:
            self.process.apply_async(
                startExperiment, args=(self.toRun[self.running],), callback=self.finished)
