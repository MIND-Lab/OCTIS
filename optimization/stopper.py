from skopt.callbacks import EarlyStopper
    
class MyCustomEarlyStopper(EarlyStopper):
    """
        Stop the optimization if the best minima
        doesn't change for n_stop iteration
    """
    def __init__(self, n_stop=10, n_random_starts=0):
        """
            Inititalize the early stopper 

            Parameters
            ----------
            n_stop : number of evaluation without improvement

        """
        #super(EarlyStopper, self).__init__()
        EarlyStopper().__init__()
        self.n_stop = n_stop
        self.n_random_starts = n_random_starts

    def convergence_res(self, res):
        """
            Given a single element of a
            Bayesian_optimization return the 
            convergence of y

            Parameters
            ----------
            res : A single element of a 
                Bayesian_optimization result

            Returns
            -------
            val : A list with the best min seen for 
                each evaluation
        """    
        val = res.func_vals
        for i in range( len(val) ):
            if( i != 0 and val[i] > val[i-1] ):
                val[i] = val[i-1]
        return val

    def _criterion(self, result):
        """
            Compute the decision to stop or not.

            Parameters
            ----------
            result : `OptimizeResult`, scipy object
                    The optimization as a OptimizeResult object.

            Returns
            -------
            decision : boolean or None
                    Return True/False if the criterion can make a decision or `None` if
                    there is not enough data yet to make a decision.

        """
        if len(result.func_vals) >= self.n_stop + self.n_random_starts:
            func_vals = self.convergence_res( result )#not sure
            #print("func_vals", func_vals)
            worst = func_vals[ len(func_vals) - (self.n_stop) ]
            best = func_vals[-1]
            #print("diff ", worst - best )
            return worst - best == 0

        else:
            return None