import logging
import time

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
from ConfigSpace.util import fix_types

class DNN(object):
    
    def __init__(self, num_layers_range:list=[1,4,10], 
                 use_dropout:bool=False, 
                 use_l2_regularization:bool=False):
        self.logger = logging.getLogger("AutoNet")
        
        self.num_layers_range = num_layers_range
    
        self.use_dropout = use_dropout
        self.use_l2_regularization = use_l2_regularization
        
        self.scalerX = StandardScaler()
        self.scalerY = StandardScaler()
    
    def fit(self, X_train, y_train, X_valid, y_valid,
            max_epochs:int,
            runcount_limit:int=100,
            wc_limit:int=60,
            config:Configuration=None,
            seed:int=12345):
        
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
                                              expected_num_epochs=max_epochs,
                                              n_outputs=1,
                                              verbose=0)
                
            history = pc.train(X_train=X_train, y_train=y_train, X_valid=X_valid,
                               y_valid=y_valid, n_epochs=1)
            
            final_loss = history["val_loss"][-1]
            
            return final_loss, {"model": pc}

        taf = SimpleTAFunc(obj_func)
        cs = ParamFCNetRegression.get_config_space(num_layers_range=self.num_layers_range,
                                                    use_l2_regularization=self.use_l2_regularization,
                                                    use_dropout=self.use_dropout)
        
        ac_scenario = Scenario({"run_obj": "quality",  # we optimize quality
                                "runcount-limit": max_epochs*runcount_limit,
                                "wallclock-limit": wc_limit,
                                "cost_for_crash": 10, 
                                "cs": cs,
                                "deterministic": "true",
                                "abort_on_first_run_crash": False,
                                "output-dir": ""
                                })
        
        intensifier = Intensifier(tae_runner=taf, stats=None,
                 traj_logger=None, 
                 rng=np.random.RandomState(42),
                 run_limit=100,
                 max_epochs=max_epochs)
        
        if isinstance(config, dict):
            config = fix_types(configuration=dict, configuration_space=cs)
            config = Configuration(configuration_space=cs, values=config)
        elif runcount_limit==1:
            config = cs.get_default_configuration()
        else:
            smac = SMAC(scenario=ac_scenario, 
                    tae_runner=taf,
                    rng=np.random.RandomState(seed),
                    intensifier=intensifier)
        
            smac.solver.runhistory.overwrite_existing_runs = True
            config = smac.optimize()
            
        
        print("Final Incumbent")
        print(config)
        
        pc = None
        start_time = time.time()
        for epoch in range(max_epochs):
            if pc is None:
                loss, model_dict = obj_func(config=config)
            else:
                loss, model_dict = obj_func(config=config, pc=pc)
            pc = model_dict["model"]
            
        print("Training Time: %d" %(time.time() - start_time))
            
        self.model = pc
    
    def predict(self, X_test):

        X_test = self.scalerX.transform(X_test)
        
        y_pred = self.model.predict(X_test)
        
        y_pred = self.scalerY.inverse_transform(y_pred)
        y_pred = 10**y_pred
        
        y_pred = np.maximum(0.0005,y_pred)
        
        return y_pred
        