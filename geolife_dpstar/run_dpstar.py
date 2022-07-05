from inspect import ArgSpec
import os.path,subprocess
from subprocess import STDOUT,PIPE
from itertools import product
import config

def run_java(setting):
    cmd = ['java', '-classpath', '"lib/**/*.jar:AdaTrace/lib/*:AdaTrace/src:AdaTrace/src/expoimpl"', 'Main']
    cmd.extend(setting)
    proc = subprocess.run(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    return proc.stdout

subprocess.run('javac -classpath "lib/**/*.jar:AdaTrace/lib/*:AdaTrace/src:AdaTrace/src/expoimpl" $(find AdaTrace/src/ -name "*.java")', shell=True)
for eps, atk in product(config.EPSILON, config.ATTACK):
    print(run_java(['/'.join([config.PREPROCESSED, "geolife_dpstar.dat"]), str(eps), str(atk), str(config.N_TRAJECTORIES)]))
