from subprocess import Popen

scens = ["SPEAR-SWV","SPEAR-IBM","CPLEX-RCW","CPLEX-REG","CPLEX-CORLAT"]
models = ["DNN", "RF"]

EVA_BUDGETs = [1,3600]
WC_BUDGET = 86400 # sec
RUNS = 3

for scen in scens:
    for model in models:
        for EVA_BUDGET in EVA_BUDGETs:
            for seed in range(1,RUNS+1):
                
                cmd = "python scripts/evaluate_matlab.py --scenario {0} --src_dir data/ --force --model {1} --budget {2} --wc_budget {3} --seed {4} 1> {0}_{1}_{2}_{3}_{4}.log 2>&1".format(
                    scen, model, EVA_BUDGET,WC_BUDGET, seed)
                
                print(cmd)
                #p = Popen(cmd, shell=True)
                #p.communicate()
        
    
    