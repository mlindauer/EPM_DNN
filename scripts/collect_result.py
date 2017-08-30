import numpy as np
import pandas as pd
from collections import OrderedDict

import tabulate
del(tabulate.LATEX_ESCAPE_RULES[u'$'])
del(tabulate.LATEX_ESCAPE_RULES[u'\\'])
del(tabulate.LATEX_ESCAPE_RULES[u'{'])
del(tabulate.LATEX_ESCAPE_RULES[u'}'])
del(tabulate.LATEX_ESCAPE_RULES[u'^'])

data = {}

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
            rmse_I = []
            rmsle_I = []
            rmse_II = []
            rmsle_II = []
            rmse_III = []
            rmsle_III = []
            rmse_IV = []
            rmsle_IV = []
            
            for seed in range(1,RUNS+1):
                with open("{0}_{1}_{2}_{3}_{4}.log".format(scen, model, EVA_BUDGET,WC_BUDGET, seed)) as fp:
                    for line in fp:
                        if line.startswith("Training Time: "):
                            t.append(int(line.split(":")[1]))
                        elif line.startswith("RMSE (I)"):
                            rmse_I.append(float(line.split(":")[1]))
                        elif line.startswith("RMSLE (I)"):
                            rmsle_I.append(float(line.split(":")[1]))
                        elif line.startswith("RMSE (II)"):
                            rmse_II.append(float(line.split(":")[1]))
                        elif line.startswith("RMSLE (II)"):
                            rmsle_II.append(float(line.split(":")[1]))
                        elif line.startswith("RMSE (III)"):
                            rmse_III.append(float(line.split(":")[1]))
                        elif line.startswith("RMSLE (III)"):
                            rmsle_III.append(float(line.split(":")[1]))
                        elif line.startswith("RMSE (IV)"):
                            rmse_IV.append(float(line.split(":")[1]))
                        elif line.startswith("RMSLE (IV)"):
                            rmsle_IV.append(float(line.split(":")[1]))
                    
            best_run = np.argmin(rmsle_valid)
                    
            data[scen][model][EVA_BUDGET]["time"] = t[best_run]
            data[scen][model][EVA_BUDGET]["RMSE (I)"] = rmse_I[best_run]
            data[scen][model][EVA_BUDGET]["RMSEL (I)"] = rmsle_I[best_run]
            data[scen][model][EVA_BUDGET]["RMSE (II)"] = rmse_II[best_run]
            data[scen][model][EVA_BUDGET]["RMSEL (II)"] = rmsle_II[best_run]
            data[scen][model][EVA_BUDGET]["RMSE (III)"] = rmse_III[best_run]
            data[scen][model][EVA_BUDGET]["RMSEL (III)"] = rmsle_III[best_run]
            data[scen][model][EVA_BUDGET]["RMSE (IV)"] = rmse_IV[best_run]
            data[scen][model][EVA_BUDGET]["RMSEL (IV)"] = rmsle_IV[best_run]
           
for budget in EVA_BUDGETs:
    table_data = [["","","\multicolumn{2}{c}{$\conf_{\text{train}}$","\multicolumn{2}{c}{$\conf_{\text{train}}$"],
                  ["Domain", "Instances","RF","DNN","RF","DNN"]
                  ]
    
    for scen in scens:
        rows = [[scen, "$\insts_{\text{train}}$", data[scen]["RF"][EVA_BUDGET]["RMSE (I)"], data[scen]["DNN"][EVA_BUDGET]["RMSE (I)"], data[scen]["RF"][EVA_BUDGET]["RMSE (III)"], data[scen]["DNN"][EVA_BUDGET]["RMSE (III)"]],
                ["", "$\insts_{\text{train}}$", data[scen]["RF"][EVA_BUDGET]["RMSE (II)"], data[scen]["DNN"][EVA_BUDGET]["RMSE (II)"], data[scen]["RF"][EVA_BUDGET]["RMSE (IV)"], data[scen]["DNN"][EVA_BUDGET]["RMSE (IV)"]]
                ]
        table_data.extend(rows)
        
    print(tabulate.tabulate(tabular_data=table_data, tablefmt="latex_booktabs"))
            
    
    
            