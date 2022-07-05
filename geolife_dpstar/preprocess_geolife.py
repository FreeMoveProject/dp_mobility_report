import csv
import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from skmob import tessellation
from skmob.tessellation import tilers

import config

# clean header of plt files and write all data into single csv
def geolife_clean_plt(root, user_id, input_filepath, traj_id):
    # read plt file
    with open(root + "/" + user_id + "/Trajectory/" + input_filepath, "rt") as fin:
        cr = csv.reader(fin)
        filecontents = [line for line in cr][6:]
        for l in filecontents:
            l.insert(0, traj_id)
            l.insert(0, user_id)
    return [filecontents[0], filecontents[-1]]

# get user id and coordinates
def geolife_data_to_dpstar(dir):
    data = []
    col_names = ["uid", "tid", "lat", "lng", "-", "Alt", "dayNo", "date", "time"]
    user_id_dirs = [
        name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))
    ]
    for user_id in np.sort(user_id_dirs):
        tempdirs = os.listdir(dir + "/" + user_id + "/Trajectory")
        subdirs = []
        for item in tempdirs:
            if not item.endswith(".DS_Store"):
                subdirs.append(item)
        traj_id = 0
        for subdir in subdirs:
            data += geolife_clean_plt(dir, user_id, subdir, traj_id)
            traj_id = traj_id + 1
    df = pd.DataFrame(data, columns=col_names)
    df['lngstring'] = df['lng'].astype(str)
    df['latstring'] = df['lat'].astype(str)
    df['locstring'] = df[['lngstring', 'latstring']].agg(','.join, axis=1)
    return df[["uid", "locstring"]]

"""# get geolife data as csv
if Path(os.path.join(config.PREPROCESSED, "geolife_dpstar.csv")).exists():
    print("Geolife csv ready")
    df = pd.read_csv(os.path.join(config.PREPROCESSED, "geolife_dpstar.csv"), columns = ["uid", "lat", "lng"])
else:
    df = geolife_data_to_dpstar(config.RAW)
    print(df.head())
    df.to_csv(os.path.join(config.PREPROCESSED, "geolife_dpstar.csv"), index=False)
    print("Geolife csv ready")"""

# get dat file for dpstar
if Path(os.path.join(config.PREPROCESSED, "geolife_dpstar.dat")).exists():
    print("Geolife dat ready")
else:
    df = geolife_data_to_dpstar(config.RAW)
    with open(os.path.join(config.PREPROCESSED, "geolife_dpstar.dat"), "a+") as f:
        users = df['uid'].unique()
        for i in range(len(users)):
            f.write("#"+str(i)+":\n")
            cond = df['uid'] == users[i]
            f.write(">0: "+ ';'.join(df[cond].locstring.values) + "\n")
        f.close()
    print("Geolife dat ready")


