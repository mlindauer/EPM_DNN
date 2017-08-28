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
    
    def fit(self, X, y,
            max_epochs:int,
            runcount_limit:int=100,
            wc_limit:int=60,
            config:Configuration=None,
            seed:int=12345):
        
        
        def obj_func(config, instance=None, seed=None, pc=None):
            # continuing training if pc is given
            # otherwise, construct new DNN
            
            models = []
            losses = []
            
            for model_idx, [train_idx, valid_idx] in enumerate([[0,3],[3,0],[1,2],[2,1]]):

                X_train = X[train_idx]
                X_valid = X[valid_idx]
                y_train = y[train_idx]
                y_valid = y[valid_idx]
                
                X_train = self.scalerX.fit_transform(X_train)
                X_valid = self.scalerX.transform(X_valid)
                
                y_train = np.log10(y_train)
                y_valid = np.log10(y_valid)
                y_train = self.scalerY.fit_transform(y_train.reshape(-1, 1))[:,0]
                y_valid = self.scalerY.transform(y_valid.reshape(-1, 1))[:,0]
                
                if pc is None:
                    
                    if model_idx == 0:
                        K.clear_session()
                    model = ParamFCNetRegression(config=config, n_feat=X_train.shape[1],
                                                 expected_num_epochs=max_epochs,
                                                 n_outputs=1,
                                                 verbose=1)
                else:
                    model = pc[model_idx]
                    
                history = model.train(X_train=X_train, y_train=y_train, X_valid=X_valid,
                                      y_valid=y_valid, n_epochs=1)
                
                models.append(model)
                
                final_loss = history["val_loss"][-1]
                losses.append(final_loss)
                
            return np.mean(losses), {"model": models}

        taf = SimpleTAFunc(obj_func)
        cs = ParamFCNetRegression.get_config_space(num_layers_range=self.num_layers_range,
                                                    use_l2_regularization=self.use_l2_regularization,
                                                    use_dropout=self.use_dropout)
        
        print(cs)
        
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
        
        
        X_all = None
        y_all = None
        for idx, (X_q, y_q) in enumerate(zip(X,y)):
            if idx == 0:
                X_all = X_q
                y_all = y_q
            else:
                X_all = np.vstack([X_all, X_q])
                y_all = np.hstack([y_all, y_q])
        
        X_all = self.scalerX.fit_transform(X_all)
        
        y_all = np.log10(y_all)
        y_all = self.scalerY.fit_transform(y_all.reshape(-1, 1))[:,0]
        
        start_time = time.time()
        
        model = ParamFCNetRegression(config=config, n_feat=X_all.shape[1],
                                         expected_num_epochs=max_epochs,
                                         n_outputs=1,
                                         verbose=1)
                  
        history = model.train(X_train=X_all, y_train=y_all, 
                              X_valid=X_all, y_valid=y_all, 
                              n_epochs=max_epochs)
            
        print("Training Time: %f" %(time.time() - start_time))
            
        self.model = model
    
    def predict(self, X_test):

        X_test = self.scalerX.transform(X_test)
        
        y_pred = self.model.predict(X_test)
        
        y_pred = self.scalerY.inverse_transform(y_pred)
        y_pred = 10**y_pred
        
        y_pred = np.maximum(0.0005,y_pred)
        
        return y_pred
        