import logging

import numpy as np

from param_net.param_fcnet import ParamFCNetRegression
from keras.losses import mean_squared_error
from keras import backend as K

from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

from mini_autonet.intensification.intensification import Intensifier
from mini_autonet.tae.simple_tae import SimpleTAFunc

from sklearn.preprocessing import StandardScaler

from ConfigSpace.configuration_space import Configuration

class DNN(object):
    
    def __init__(self, max_layers:int=10, 
                 use_dropout:bool=False, 
                 use_l2_regularization:bool=False):
        self.logger = logging.getLogger("AutoNet")
        
        self.max_layers = max_layers
    
        self.use_dropout = use_dropout
        self.use_l2_regularization = use_l2_regularization
        
        self.scalerX = StandardScaler()
        self.scalerY = StandardScaler()
    
    def fit(self, X_train, y_train, X_valid, y_valid,
            max_epochs:int,
            runcount_limit:int=100,
            config:Configuration=None):
        
        X_train = self.scalerX.fit_transform(X_train)
        X_valid = self.scalerX.transform(X_valid)
        
        y_train = np.log10(y_train)
        y_valid = np.log10(y_valid)
        y_train = self.scalerY.fit_transform(y_train)
        y_valid = self.scalerY.transform(y_valid)
        
        
        def obj_func(config, instance=None, seed=None, pc=None):
            # continuing training if pc is given
            # otherwise, construct new DNN
            if pc is None:
                K.clear_session()
                pc = ParamFCNetRegression(config=config, n_feat=X_train.shape[1],
                                              max_num_epochs=max_epochs,
                                              n_outputs=1,
                                              verbose=1)
                
            history = pc.train(X_train=X_train, y_train=y_train, X_valid=X_valid,
                               y_valid=y_valid, n_epochs=1)
            
            final_loss = history.history["val_loss"][-1]
            
            return final_loss, {"model": pc}

        taf = SimpleTAFunc(obj_func)
        cs = ParamFCNetRegression.get_config_space(max_num_layers=self.max_layers,
                                                    use_l2_regularization=self.use_l2_regularization,
                                                    use_dropout=self.use_dropout)
        
        ac_scenario = Scenario({"run_obj": "quality",  # we optimize quality
                                "runcount-limit": max_epochs*runcount_limit,
                                "cost_for_crash": 10, 
                                "cs": cs,
                                "deterministic": "true",
                                "output-dir": ""
                                })
        
        intensifier = Intensifier(tae_runner=taf, stats=None,
                 traj_logger=None, 
                 rng=np.random.RandomState(42),
                 run_limit=100,
                 max_epochs=max_epochs)
        
        if isinstance(config, dict):
            config = Configuration(configuration_space=cs, values=config)
        elif runcount_limit==1:
            config = cs.get_default_configuration()
        else:
            smac = SMAC(scenario=ac_scenario, 
                    tae_runner=taf,
                    rng=np.random.RandomState(42),
                    intensifier=intensifier)
        
            smac.solver.runhistory.overwrite_existing_runs = True
            config = smac.optimize()
            
        
        print("Final Incumbent")
        print(config)
        
        pc = None
        for epoch in range(max_epochs):
            if pc is None:
                loss, model_dict = obj_func(config=config)
            else:
                loss, model_dict = obj_func(config=config, pc=pc)
            pc = model_dict["model"]
            
        self.model = pc
    
    def predict(self, X_test):

        X_test = self.scalerX.transform(X_test)
        
        y_pred = self.model.predict(X_test)
        
        y_pred = self.scalerY.inverse_transform(y_pred)
        y_pred = 10**y_pred
        
        return y_pred
        