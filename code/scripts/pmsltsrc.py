import pandas as pd
import numpy as np
import numba

@numba.jit
def get_r(apc, rates, t):
    return np.power(rates * (1 + apc), (t - 2021))

class pmsltModel:
    """
    This class implements the PMSLT model for smoking and vaping prevalence
    """
    def __init__(self, timesteps: int = 4, sex: str = "Male") -> None:
        """
        Initializes the PMSLT model with the given initial and target prevalence data.

        Also creates a set up flows, states, and flow lookup dictionary.
        """        
        self.initPrev = f"../data/initialPrevalence{sex}.csv"
        self.prevState = pd.read_csv(self.initPrev)
        self.sex = sex.lower()
        self.flows = ['n_s', 's_rs', 'n_v', 'v_rv', 'rs_dead', 'rv_dead',
       's_dead', 'sv_dead', 'v_dead', 'n_dead', 'v_sv', 's_sv', 'sv_s',
       'sv_vrs', 'sv_rs', 'v_s', 's_vrs', 'vrs_sv', 'vrs_s', 'vrs_rv',
       'vrs_dead']
        
        self.ps = ['p_n_s', 'p_s_rs', 'p_n_v', 'p_v_rv', 'p_rs_dead', 'p_rv_dead', 'p_s_dead', 'p_sv_dead', 'p_v_dead', 'p_n_dead', 'p_v_sv', 'p_s_sv', 'p_sv_s', 'p_sv_vrs', 'p_sv_rs', 'p_v_s', 'p_s_vrs', 'p_vrs_sv', 'p_vrs_s', 'p_vrs_rv', 'p_vrs_dead']
        self.x = set(flow.split('_')[0] for flow in self.flows)
        self.states = set(flow.split('_')[1] for flow in self.flows)

        self.modelCols = ['age', 'sex', 's', 'v', 'rs', 'rv', 'dead', 'vrs', 'sv', 'n']
        
        self.flowsLookup = {'rs': ['rs_dead'],
                            'sv': ['sv_dead', 'sv_s', 'sv_vrs', 'sv_rs'],
                            's': ['s_rs', 's_dead', 's_sv', 's_vrs'],
                            'vrs': ['vrs_sv', 'vrs_s', 'vrs_rv', 'vrs_dead'],
                            'v': ['v_rv', 'v_dead', 'v_sv', 'v_s'],
                            'rv': ['rv_dead'],
                            'n': ['n_s', 'n_v', 'n_dead']}
        
        self.s = timesteps
        
        self.targets = pd.read_csv(f"../data/prevalenceTargets{sex}.csv")
        self.dictStates = {}

    def se_score(self, outputs: pd.DataFrame, year: int) -> float:
        """
        Calculates the mean squared error between the model outputs and the target prevalence data for a given year.

        Currently unweighted.
        """
        targets = self.targets[self.targets['year'] == year]
        score = np.sum(np.square(outputs['smoking'] - targets['smoking']) + np.square(outputs['vaping'] - targets['vaping']))
        return score

    def pmslt(self, rates, apc, t1 = 2021, t2 = 2039) -> pd.DataFrame:
        """
        The core PMSLT model implementation. Takes in flow rates and APCs and returns the performance score of the model.

        The functions follow the PMSLT model equations as described in the problem paper.
        """  
        if np.shape(rates) != np.shape(apc): # making sure the dimensions are same (To allow for matrix multiplication)
            print("Error: flow rates and flow apc must have the same shape")
            return
        
        self.prevState = pd.read_csv(self.initPrev) # reset the model to the initial state
        score = 0

        for t in range(t1, t2 + 1): # loop through the years

            self.dictStates[t] = self.prevState.copy()
            #r = np.power(rates * (1 + apc), (t - 2021))
            r = pd.DataFrame(get_r(rates, apc, t), columns= self.flows)

            for statex in self.x:
                r[f"f_{statex}"] = r[self.flowsLookup[statex]].sum(axis=1)

            for flow in self.flows:
                statex = flow.split('_')[0]
                statey = flow.split('_')[1]
                r[f"p_{statex}_{statey}"] = (1 - np.exp( - r[f"f_{statex}"]/self.s)) * r[f"{statex}_{statey}"]/r[f"f_{statex}"]

            for state in self.x:
                r[f"p_from_{state}"] = r[[col for col in self.ps if col.endswith(f"_{state}")]].sum(axis=1)

            prevModel = self.prevState.copy()

            for i in range(self.s):
                for state in self.states:
                    if state != 'dead':
                        prevModel[f"out_{state}"] = r[f"p_from_{state}"] * prevModel[state]
                    else:
                        prevModel[f"out_{state}"] = 0
                        
                    prevModel[f"in_{state}"] = 0
                    for statex in self.x:
                        if statex != state:
                            if f"p_{statex}_{state}" in r.columns:
                                prevModel[f"in_{state}"] += r[f"p_{statex}_{state}"] * prevModel[statex]

                    prevModel[state] += prevModel[f"in_{state}"] - prevModel[f"out_{state}"]

            self.prevState = prevModel[self.modelCols]

            prevModel["smoking"] = (prevModel["s"] + prevModel["sv"])/(1 - prevModel["dead"])
            prevModel["vaping"] = (prevModel["v"] + prevModel["sv"] + prevModel["vrs"])/(1 - prevModel["dead"])

            score += self.se_score(prevModel[["age", "sex", "smoking", "vaping"]], t)

        return score, prevModel


        

