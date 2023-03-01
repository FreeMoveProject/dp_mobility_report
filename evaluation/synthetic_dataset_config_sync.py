import evaluation_config
import os

# Configure dpstar config file
with open(os.path.join(evaluation_config.PATH_DPSTAR, "dpstar_config.py"), "w") as f:
    f.write("RAW = '%s/%s' \nPREPROCESSED_DIR = '%s' \nPREPROCESSED_FILE = '%s' \nUSER_ID = 'uid' \nLATITUDE = 'lat' \nLONGITUDE = 'lng' \nEPSILON "
            "= [%s] \nATTACK = [False] \nN_TRAJECTORIES = %s \nRESULTS = '%s'" % (
                evaluation_config.PATH_RAW_DATA_DIR, evaluation_config.RAW_DATA_FILE_PEKING, evaluation_config.PATH_DPSTAR_PREPROCESSED_DIR,
                evaluation_config.PATH_DPSTAR_PREPROCESSED_FILE,
                evaluation_config.EPSILON, evaluation_config.NUMBER_OF_RUNS, evaluation_config.PATH_DPSTAR_RESULT_DIR))

# NOW RUN THE "run_dpstar.bash" IN THE FITTING CONDA INTERPRETER

# subprocess.run(["cd %s; bash run_dpstar.bash" % evaluation_config.PATH_DPSTAR], shell=True)
