import os
import time
import json
from pathlib import Path
from collections import namedtuple
from octis.dashboard.experimentManager import startExperiment
import octis.dashboard.experimentManager as expManager
import multiprocessing as mp
from subprocess import Popen
import signal


class QueueManager:
    """
    The QueueManager class is used to track old, ongoing and new experiments
    """
    running = None
    toRun = None
    order = None
    completed = None
    process = None
    busy = None
    idle = None

    def __init__(self):
        """
        Initialize the queue manager.
        Loads old queues
        """
        manager = mp.Manager()
        self.running = manager.list()
        self.running.append(None)
        self.toRun = manager.dict()
        self.order = manager.list()
        self.completed = manager.dict()
        self.busy = manager.list()
        self.busy.append(False)
        self.process = manager.list()

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

        :return: a tuple containing id of the batch and id of
                 the next experiment to run
        :rtype: tuple
        """
        if self.running[0] is None:
            self.running[0] = self.order.pop(0)
            self.busy[0] = False
            self.start()
        return self.running[0]

    def add_experiment(self, batch, id, parameters):
        """
        Adds a new experiment to the queue

        :param batch: id of the batch
        :type batch: String
        :param id: id of the experiment
        :type id: String
        :param parameters: dictionary with the parameters of the experiment
        :type parameters: Dict

        :return: True if the experiment was added to the queue, False otherwise
        :rtype: boolean
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

        :return: a tuple containing id of the batch and id of the
                 completed experiment
        :rtype: tuple
        """
        while True:
            time.sleep(4)
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
            time.sleep(6)

    def pause(self):
        """
        pause the running experiment

        :return: a tuple containing id of the batch and id of the
                 paused experiment
        :rtype: tuple
        """
        if self.busy[0] and self.running[0] != None:
            paused = self.running[0]
            to_stop = self.process.pop()
            os.kill(to_stop, signal.SIGTERM)
            self.order.insert(0, paused)
            self.running[0] = None
            return paused
        return False

    def getBatchNames(self):
        """
        Get the name of each batch with experiment in the completed list or
        in the list of experiments to run

        :return: names of each batch
        :rtype: List
        """
        batch_names = []
        to_remove = []
        for key, value in self.completed.items():
            if not self.getExperimentInfo(value["batchId"], value["experimentId"]):
                to_remove.append(key)
            else:
                if value["batchId"] not in batch_names:
                    batch_names.append(value["batchId"])
        for el in to_remove:
            del self.completed[el]

        for key, value in self.toRun.items():
            if value["batchId"] not in batch_names:
                batch_names.append(value["batchId"])
        return batch_names

    def getBatchExperiments(self, batch_name):
        """
        Retrieves all the experiments of the selected batch

        :param batch_name: name of the batch
        :type batch_name: String

        :return: list of experiments metadata
        :rtype: List
        """
        experiments = []
        to_remove = []
        for key, value in self.completed.items():
            if not self.getExperimentInfo(value["batchId"], value["experimentId"]):
                to_remove.append(key)
            else:
                if value["batchId"] == batch_name:
                    experiments.append(value)
        for el in to_remove:
            del self.completed[el]

        for key, value in self.toRun.items():
            if value["batchId"] == batch_name:
                experiments.append(value)
        return experiments

    def getExperimentInfo(self, batch, experimentId):
        """
        Return the info of the experiment with the given batch name and id

        :param batch: name of the batch
        :type batch: String
        :param experimentId: name of the experiment
        :type experimentId: String

        :return: experiment info (mean, median, best seen, worst seen)
        :rtype: Dict
        """
        experiment = None
        if batch + experimentId in self.completed:
            experiment = self.completed[batch+experimentId]
        if batch + experimentId in self.toRun:
            experiment = self.toRun[batch+experimentId]
        if experiment is not None:
            path = str(os.path.join(
                experiment["path"], experiment["experimentId"], experiment["experimentId"]+".json"))
            return expManager.singleInfo(path)
        return None

    def start(self):
        """
        Start a new experiment in a new process
        """
        if not self.busy[0]:
            self.busy[0] = True
            process = mp.Process(
                target=QueueManager._execute_and_update, args=(
                    self.toRun,
                    self.running,
                    self.busy
                ))
            process.start()
            print("starting "+self.running[0])
            self.process.append(process.pid)

    def stop(self):
        """
        Stop the current experiment and save the information about it
        """
        self.idle.terminate()
        self.pause()
        self.save_state()

    @staticmethod
    def _execute_and_update(toRun, running, busy):
        """
        start an experiment using a static method

        :param toRun: list of experiments to run
        :type: List
        :param running: id of the running experiment
        :type: String
        :param busy: specify if the queuemanager is busy
        :type busy: boolean
        """
        startExperiment(toRun[running[0]])
        busy[0] = False

    def getModel(self, batch, experimentId, iteration, modelRun):
        """
        Retrieve output of the model for a single model

        :param batch: name of the batch
        :type batch: String
        :param experimentId: name of the experiment
        :type experimentId: String
        :param iterarion: number of iteration of the model to retrieve
        :type iteration: Int
        :param modelRun: numeber of model run of the model to retrieve
        :type modelRun: Int

        :return: output of the model (topic-word-matrix,
                 document-topic-matrix and vocabulary)
        :rtype: Dict
        """
        experiment = None
        if batch+experimentId in self.completed:
            experiment = self.completed[batch+experimentId]
        if batch+experimentId in self.toRun:
            experiment = self.toRun[batch+experimentId]
        if experiment is not None:
            path = str(os.path.join(
                experiment["path"], experiment["experimentId"]))
            return expManager.getModelInfo(path, iteration, modelRun)
        return False

    def getExperimentIterationInfo(self, batch, experimentId, iteration=0):
        """
        Retrieve the results of the BO untile the given iteration

        :param batch: id of the batch
        :type batch: String
        :param experimentId: id of the experiment
        :type experimentId: String
        :param iteration: last iteration to consider
        :type iteration: Int

        :return: experiment interation info
        :rtype: Dict
        """
        experiment = None
        if batch+experimentId in self.completed:
            experiment = self.completed[batch+experimentId]
        if batch+experimentId in self.toRun:
            experiment = self.toRun[batch+experimentId]
        if experiment is not None:
            path = str(os.path.join(
                experiment["path"], experiment["experimentId"], experiment["experimentId"]+".json"))
            return expManager.retrieveIterationBoResults(path, iteration)
        return False

    def getExperiment(self, batch, experimentId):
        """
        Retrieve metadata about the experiment

        :param batch: name of the batch
        :type batch: String
        :param experiment: name of the experiment
        :type experiment: String

        :return: return experimen metadatata
        :rtype: Dict
        """
        experiment = False
        if batch+experimentId in self.completed:
            experiment = self.completed[batch+experimentId]
        if batch+experimentId in self.toRun:
            experiment = self.toRun[batch+experimentId]
        return experiment

    def getAllExpIds(self):
        """
        Retrieve the name of each experiment and their batch

        :param expIds: list of entries. Each entry is a list with 2 elements.
                 the name of the experiment and a list with name of the batch and
                 name of the experiment
        :type expIds: List

        :return: name and batch of each experiment
        :rtype: List
        """
        expIds = []
        to_remove = []
        for key, exp in self.completed.items():
            if not self.getExperimentInfo(exp["batchId"], exp["experimentId"]):
                to_remove.append(key)
            else:
                expIds.append([exp["experimentId"],
                               [exp["batchId"], exp["experimentId"]]])
        for el in to_remove:
            del self.completed[el]
        for key, exp in self.toRun.items():
            expIds.append([exp["experimentId"],
                           [exp["batchId"], exp["experimentId"]]])
        return expIds

    def getToRun(self):
        """
        Retrieve the experiments to run

        :return: dictionary of the experiments to run
        :rtype: Dict
        """
        return dict(self.toRun)

    def getOrder(self):
        """
        Retrieve the order of the experiments to run

        :return: id of each experiment to run in order (from first to last)
        :rtype: List
        """
        return list(self.order)

    def getRunning(self):
        """
        returns the id of the running experiment

        :return: id of the running experiment
        :rtype: String
        """
        return self.running[0]

    def editOrder(self, newOrder):
        """
        Updates the order of the experiments to run

        :param newOrder: new order of the experiments
        :type newOrder: List
        """
        FinalOrder = []
        for el in newOrder:
            if el in self.order:
                FinalOrder.append(el)
        self.order[:] = []
        self.order.extend(FinalOrder)

    def deleteFromOrder(self, experimentId):
        """
        Delete and experiment from the queue of experiments to run

        :param experimentId: id of the experiment to remove from the queue
        :type experimentId: String
        """
        self.order = list(filter(lambda a: a != experimentId, self.order))
        del self.toRun[experimentId]
