######################################################################
import logging
import sys
import os
import inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0,cmd_folder)
    
######################################################################

import os
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from ConfigSpace.io import pcs
from ConfigSpace.util import fix_types, deactivate_inactive_hyperparameters
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from smac.configspace.util import convert_configurations_to_array

from epm_dnn.dnn import DNN

def read_feature_file(fn:str):
    
    return pd.read_csv(fn, header=0, index_col=0)

def read_perf_file(fn:str):
    
    perf_pd = pd.read_csv(fn, header=None, index_col=0)
    perf_pd.replace(0.0, 0.0005, inplace=True)
    return perf_pd

def read_config_file(fn:str, cs:ConfigurationSpace):
    
    config_pd = pd.read_csv(fn, header=0, index_col=0, dtype=object)    
    
    configs = []
    
    for param_name in list(config_pd):
        if param_name.startswith("dummy_non_parameter"):
            del config_pd[param_name]
            
    for config in config_pd.iterrows():
        config = fix_types(configuration=config[1:][0].to_dict(), configuration_space=cs)
        config = deactivate_inactive_hyperparameters(configuration=config, configuration_space=cs)
        configs.append(config)
        
    return configs

def read_cs(fn:str):

    with open(fn) as fp:
        pcs_str = fp.readlines()
        cs = pcs.read(pcs_str)
        
    return cs

def build_matrix(feature_pd:pd.DataFrame, perf_pd:pd.DataFrame, 
                 configs:list, cs:ConfigurationSpace,
                 n_insts:int=None):
    
    insts = list(feature_pd.index)
    
    if n_insts is not None and n_insts < len(insts):
        insts = random.sample(insts, n_insts)
        
    config_matrix = convert_configurations_to_array(configs)
    
    # one hot encode categorical parameters
    n_values = []
    mask_array = []
    parameters = cs.get_hyperparameters()
    
    for param in parameters:
        if isinstance(param, (CategoricalHyperparameter)):
            n_values.append(len(param.choices))
            mask_array.append(True)
        else:
            mask_array.append(False)
    
    n_values = np.array(n_values)
    mask_array = np.array(mask_array)        
    
    ohe = OneHotEncoder(n_values=n_values, categorical_features=mask_array, sparse=False)
    config_matrix = ohe.fit_transform(config_matrix)
    
    # convert in X matrix and y vector
    X = []
    y = []
    for inst in insts:
        feat_vector = feature_pd.loc[inst].values
        perf_vector = perf_pd.loc[inst].values
        for c_idx in range(len(configs)):
            config_vec = config_matrix[c_idx,:]
            perf = perf_vector[c_idx]
            
            X.append(np.concatenate((config_vec, feat_vector)))
            y.append(perf)
    
    X = np.array(X)
    y = np.array(y)
    
    print(X.shape)
    print(y.shape)
    
    return X, y
        
if __name__ == "__main__":
    
    random.seed(12345)
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--n_insts", type=int, default=None, 
                        help="subsample to x instances")
    parser.add_argument("--force_reading", default=False,
                        action="store_true")
    
    parser.add_argument("--verbose", choices=["INFO","DEBUG"], default="INFO")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=args.verbose)
    
    if args.scenario == 'SPEAR-SWV':
        performance_file = os.path.join(args.src_dir,'SAT','1000samples-SPEAR-SWV-all604inst-results.txt')
        config_file = os.path.join(args.src_dir,'SAT','1000samples-algospear1.2.1.1-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(args.src_dir,'SAT','spear-params.txt')
        feature_file = os.path.join(args.src_dir,'SAT','SWV-feat.csv')
    elif args.scenario == 'SPEAR-IBM':
        performance_file = os.path.join(args.src_dir,'SAT','1000samples-SPEAR-IBM-all765inst-results.txt')
        config_file = os.path.join(args.src_dir,'SAT','1000samples-algospear1.2.1.1-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(args.src_dir,'SAT','spear-params.txt')
        feature_file = os.path.join(args.src_dir,'SAT','IBM-ALL-feat.csv')
    elif args.scenario == 'SPEAR-SWV-IBM':
        performance_file = os.path.join(args.src_dir,'SAT','1000samples-SPEAR-IBM-SWV-results.txt')
        config_file = os.path.join(args.src_dir,'SAT','1000samples-algospear1.2.1.1-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(args.src_dir,'SAT','spear-params.txt')
        feature_file = os.path.join(args.src_dir,'SAT','IBM-SWV-feat.csv')
    elif args.scenario == 'CPLEX-CRR':
        tuning_scenario = 'CPLEX-CRR' #'CPLEX12-cat-CORLAT-REG-RCW' # memory error
        performance_file = os.path.join(args.src_dir,'MIP','1000samples-CPLEX-CORLAT-REG-RCW-results.txt')
        config_file = os.path.join(args.src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(args.src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
        feature_file = os.path.join(args.src_dir,'MIP','CORLAT-REG-RCW-features.csv')
    elif args.scenario == 'CPLEX-CR':
        performance_file = os.path.join(args.src_dir,'MIP','1000samples-CPLEX-CORLAT-REG-results.txt')
        config_file = os.path.join(args.src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(args.src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
        feature_file = os.path.join(args.src_dir,'MIP','CORLAT-REG-features.csv')
    elif args.scenario == 'CPLEX-RCW':
        performance_file = os.path.join(args.src_dir,'MIP','1000samples-CPLEX-RCW-990train-990test-results.txt')
        config_file = os.path.join(args.src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(args.src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
        feature_file = os.path.join(args.src_dir,'MIP','RCW-train_test-features-withfilename.csv')
    elif args.scenario == 'CPLEX-REG':
        performance_file = os.path.join(args.src_dir,'MIP','1000samples-CPLEX-CATS_REG-1000train-1000test-results.txt')
        config_file = os.path.join(args.src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(args.src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
        feature_file = os.path.join(args.src_dir,'MIP','REG-train_test-features-withfilename.csv')
    elif args.scenario == 'CPLEX-CORLAT':
        performance_file = os.path.join(args.src_dir,'MIP','1000samples-CPLEX-CORLAT-train_test_inst-results.txt')
        config_file = os.path.join(args.src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(args.src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
        feature_file = os.path.join(args.src_dir,'MIP','CORLAT-train_test-features-withfilename.csv')
    elif args.scenario == 'CPLEX-BIGMIX':
        performance_file = os.path.join(args.src_dir,'MIP','1000samples-CPLEX-BIGMIX-all1510inst-results.txt')
        config_file = os.path.join(args.src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(args.src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
        feature_file = os.path.join(args.src_dir,'MIP','BIGMIX-train_test-features-withfilename.csv')
    
    
    if os.path.isfile("converted_data/%s/X.npy" %(args.scenario)) and not args.force_reading:
        X = np.load("converted_data/%s/X.npy" %(args.scenario))
        y = np.load("converted_data/%s/y.npy" %(args.scenario))
    
    else:
        feature_pd = read_feature_file(fn=feature_file)
        cs = read_cs(fn=pcs_file)
        perf_pd = read_perf_file(fn=performance_file)
        print(perf_pd.min().min())
        configs = read_config_file(fn=config_file, cs=cs)
        
        X, y = build_matrix(feature_pd=feature_pd, 
                             perf_pd=perf_pd, 
                             configs=configs, 
                             cs=cs, 
                             n_insts=args.n_insts)
        
        try:
            os.makedirs("converted_data/%s/"%(args.scenario))
        except OSError:
            pass
        
        np.save(file="converted_data/%s/X.npy" %(args.scenario), 
                arr=X)
        np.save(file="converted_data/%s/y.npy" %(args.scenario), 
                arr=y)
    
    dnn = DNN(max_layers=10, 
              use_dropout=False, 
              use_l2_regularization=False)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=12345)
    
    dnn.fit(X_train=X_train, 
            y_train=y_train,
            X_valid=X_valid, 
            y_valid=y_valid,
            max_epochs=10,
            runcount_limit=10)
    
    y_pred = dnn.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
    
    print("RMSE (test): %f" %(rmse))
    