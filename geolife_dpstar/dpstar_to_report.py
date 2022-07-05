import glob
import config
import os
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from skmob import tessellation
from skmob.tessellation import tilers

syn_dpstar = [f for f in os.listdir(config.PREPROCESSED) if 'eps' in f]

for filename in syn_dpstar:
    with open(os.path.join(config.PREPROCESSED, filename), "r") as f:
        lines = f.readlines()
        to_write = []
        uid = 0
        for i in range(1, len(lines), 2):
            locations = lines[i][3:].split(';')
            to_write.extend([','.join([str(uid),','.join(list(reversed(loc.split(','))))]) for loc in locations[:-1]])
            uid = uid+1
        f.close()
    with open(os.path.join(config.RESULTS, filename[:-4]+'.csv'), "w+") as f:
        f.write('uid,lat,lng\n'+'\n'.join(to_write))
    f.close()
    df = pd.read_csv(os.path.join(config.RESULTS, filename[:-4]+'.csv'), header=0)

