import logging
import time

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

from mini_autonet.intensification.intensification import Intensifier
from mini_autonet.tae.simple_tae import SimpleTAFunc

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant
from ConfigSpace.util import fix_types

class RF(object):
    
    def __init__(self):
        self.logger = logging.getLogger("RF")
        
    def fit(self, X_train, y_train, X_valid, y_valid,
            runcount_limit:int=100,
            wc_limit:int=60,
            config:Configuration=None,
            seed:int=12345):
        
        y_train = np.log10(y_train)
        y_valid = np.log10(y_valid)
        
        def obj_func(config, instance=None, seed=None, pc=None):
            rf = RandomForestRegressor(n_estimators=config["n_estimators"], 
                                  criterion=config["criterion"], 
                                  min_samples_split=config["min_samples_split"], 
                                  min_samples_leaf=config["min_samples_leaf"], 
                                  min_weight_fraction_leaf=config["min_weight_fraction_leaf"], 
                                  max_features=config["max_features"],
                                  bootstrap=config["bootstrap"], 
                                  random_state=12345)
            
            rf.fit(X_train, y_train)
            
            y_preds = []
            for tree in rf.estimators_:
                y_pred = 10**tree.predict(X_valid)
                y_preds.append(y_pred)
            y_preds = np.mean(y_preds, axis=0)
            y_preds = np.log10(y_preds)
                
            return np.sqrt(mean_squared_error(y_true=y_valid, y_pred=y_preds))

        taf = SimpleTAFunc(obj_func)
        cs = self.get_config_space()
        
        ac_scenario = Scenario({"run_obj": "quality",  # we optimize quality
                                "runcount-limit": runcount_limit,
                                "wallclock-limit": wc_limit,
                                "cost_for_crash": 10, 
                                "cs": cs,
                                "deterministic": "true",
                                "output-dir": ""
                                })
        
        if isinstance(config, dict):
            config = fix_types(configuration=dict, configuration_space=cs)
            config = Configuration(configuration_space=cs, values=config)
        elif runcount_limit==1:
            config = cs.get_default_configuration()
        else:
            smac = SMAC(scenario=ac_scenario, 
                    tae_runner=taf,
                    rng=np.random.RandomState(seed))
            config = smac.optimize()
        
        print("Final Incumbent")
        print(config)
        
        rf = RandomForestRegressor(n_estimators=100, 
                                  criterion=config["criterion"], 
                                  min_samples_split=config["min_samples_split"], 
                                  min_samples_leaf=config["min_samples_leaf"], 
                                  min_weight_fraction_leaf=config["min_weight_fraction_leaf"], 
                                  max_features=config["max_features"],
                                  bootstrap=config["bootstrap"], 
                                  random_state=12345)
            
        start_time = time.time()
        rf.fit(X_train, y_train)
        print("Training Time: %d" %(time.time() - start_time))
            
        self.model = rf
    
    def predict(self, X_test):

        y_preds = []
        for tree in self.model.estimators_:
            y_pred = 10**tree.predict(X_test)
            y_preds.append(y_pred)
        y_preds = np.mean(y_preds, axis=0)
        
        y_pred = np.maximum(0.0005,y_pred)
        
        return y_pred
    
    def get_config_space(self):
        # copied from Auto-Sklearn
        
        cs = ConfigurationSpace()
        n_estimators = Constant("n_estimators", 10)
        criterion = Constant("criterion", "mse")
        max_features = UniformFloatHyperparameter(
            "max_features", 0.5, 1, default=1)
        max_depth = UnParametrizedHyperparameter("max_depth", "None")
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default=1)
        min_weight_fraction_leaf = \
            UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default="True")

        cs.add_hyperparameters([n_estimators, criterion, max_features,
                                max_depth, min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes,
                                bootstrap])
        
        return cs
        