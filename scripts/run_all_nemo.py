from subprocess import Popen

scens = ["SPEAR-SWV","SPEAR-IBM","CPLEX-RCW","CPLEX-REG","CPLEX-CORLAT"]
models = ["DNN", "RF"]

EVA_BUDGET = 1 #3600
WC_BUDGET = 86400 # sec

for scen in scens:
    for model in models:
    
        cmd = "python scripts/evaluate_matlab.py --scenario {0} --src_dir data/ --force --model {1} --budget {2} --wc_budget {3} 1> {0}_{1}_{2}_{3}.log 2>&1".format(
            scen, model, EVA_BUDGET,WC_BUDGET)
        
        print(cmd)
        #p = Popen(cmd, shell=True)
        #p.communicate()
        
    
    