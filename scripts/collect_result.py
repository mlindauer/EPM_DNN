import numpy as np
import pandas as pd
from _collections import OrderedDict

data = {}
flat_data = OrderedDict() 

scens = ["SPEAR-SWV","SPEAR-IBM","CPLEX-RCW","CPLEX-REG","CPLEX-CORLAT"]
models = ["DNN", "RF"]

EVA_BUDGETs = [1]#,3600]
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
                    
            data[scen][model][EVA_BUDGET]["time"] = np.median(t)
            data[scen][model][EVA_BUDGET]["RMSE (train)"] = np.median(rmse_train)
            data[scen][model][EVA_BUDGET]["RMSEL (train)"] = np.median(rmsle_train)
            data[scen][model][EVA_BUDGET]["RMSE (valid)"] = np.median(rmse_valid)
            data[scen][model][EVA_BUDGET]["RMSEL (valid)"] = np.median(rmsle_valid)
            data[scen][model][EVA_BUDGET]["RMSE (test)"] = np.median(rmse_test)
            data[scen][model][EVA_BUDGET]["RMSEL (test)"] = np.median(rmsle_test)
           
            key = "{0}_{1}_{2}".format(scen, model, EVA_BUDGET)
            flat_data[key] = flat_data.get(key,OrderedDict())
            flat_data[key]["time [sec]"] = np.median(t)
            flat_data[key]["RMSE (train)"] = np.median(rmse_train)
            flat_data[key]["RMSEL (train)"] = np.median(rmsle_train)
            flat_data[key]["RMSE (valid)"] = np.median(rmse_valid)
            flat_data[key]["RMSEL (valid)"] = np.median(rmsle_valid)
            flat_data[key]["RMSE (test)"] = np.median(rmse_test)
            flat_data[key]["RMSEL (test)"] = np.median(rmsle_test)
             
            
df = pd.DataFrame.from_dict(data=flat_data)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)


            