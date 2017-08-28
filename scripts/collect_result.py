import numpy as np
import pandas as pd
from _collections import OrderedDict

data = {}
flat_data = OrderedDict() 

scens = ["SPEAR-SWV","SPEAR-IBM","CPLEX-RCW","CPLEX-REG","CPLEX-CORLAT"]
models = ["DNN", "RF"]

EVA_BUDGETs = [1,3600]
WC_BUDGET = 86400 # sec
RUNS = 3

for scen in scens:
    for model in models:
        for EVA_BUDGET in EVA_BUDGETs:
            
            data[scen] = data.get(scen,{})
            data[scen][model] = data[scen].get(model,{})
            data[scen][model][EVA_BUDGET] = data[scen][model].get(EVA_BUDGET,{})

            t = []
            rmse_train = []
            rmsle_train = []
            rmse_valid = []
            rmsle_valid = []
            rmse_test = []
            rmsle_test = []
            
            for seed in range(1,RUNS+1):
                with open("{0}_{1}_{2}_{3}_{4}.log".format(scen, model, EVA_BUDGET,WC_BUDGET, seed)) as fp:
                    for line in fp:
                        if line.startswith("Training Time: "):
                            t.append(int(line.split(":")[1]))
                        elif line.startswith("RMSE (train)"):
                            rmse_train.append(float(line.split(":")[1]))
                        elif line.startswith("RMSLE (train)"):
                            rmsle_train.append(float(line.split(":")[1]))
                        elif line.startswith("RMSE (valid)"):
                            rmse_valid.append(float(line.split(":")[1]))
                        elif line.startswith("RMSLE (valid)"):
                            rmsle_valid.append(float(line.split(":")[1]))
                        elif line.startswith("RMSE (test)"):
                            rmse_test.append(float(line.split(":")[1]))
                        elif line.startswith("RMSLE (test)"):
                            rmsle_test.append(float(line.split(":")[1]))        
                    
            median_run = np.argsort(rmsle_valid)[len(rmsle_valid)//2]
                    
            data[scen][model][EVA_BUDGET]["time"] = t[median_run]
            data[scen][model][EVA_BUDGET]["RMSE (train)"] = rmse_train[median_run]
            data[scen][model][EVA_BUDGET]["RMSEL (train)"] = rmsle_train[median_run]
            data[scen][model][EVA_BUDGET]["RMSE (valid)"] = rmse_valid[median_run]
            data[scen][model][EVA_BUDGET]["RMSEL (valid)"] = rmsle_valid[median_run]
            data[scen][model][EVA_BUDGET]["RMSE (test)"] = rmse_test[median_run]
            data[scen][model][EVA_BUDGET]["RMSEL (test)"] = rmsle_test[median_run]
           
            key = "{0}_{1}_{2}_{3}".format(scen, model, EVA_BUDGET, median_run)
            flat_data[key] = flat_data.get(key,OrderedDict())
            flat_data[key]["time [sec]"] = t[median_run]
            flat_data[key]["RMSE (train)"] = rmse_train[median_run]
            flat_data[key]["RMSEL (train)"] = rmsle_train[median_run]
            flat_data[key]["RMSE (valid)"] = rmse_valid[median_run]
            flat_data[key]["RMSEL (valid)"] = rmsle_valid[median_run]
            flat_data[key]["RMSE (test)"] = rmse_test[median_run]
            flat_data[key]["RMSEL (test)"] = rmsle_test[median_run]
             
            
df = pd.DataFrame.from_dict(data=flat_data)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)

df.to_csv("detail_results.csv")

            