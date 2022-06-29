import os
import pandas as pd 
import numpy as np 
import json 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing

def load_data():
    main_dir = "./data/WSDA_IR_Soil/"
    years = sorted(os.listdir(main_dir), key=lambda x: int(x.split("_")[-1].split(".")[0]))[3:10]

    farms = {}
    ids = set()
    dfs = []

    converted_dataset = [] 


    cats = [
        'CropGroup',
        'CropType',
        'TotalAcres',
        'Shape_Leng', 
        'lat', 
        'lon', 
        'watershed', 
        'Shape_Le_1', 
        'Shape_Area', 
        'slopegradw', 
        'wtdepannmi', 
        'wtdepaprju', 
        'flodfreqdc', 
        'flodfreqma', 
        'pondfreqpr', 
        'aws025wta', 
        'aws0100wta', 
        'aws0150wta', 
        'drclassdcd', 
        'drclasswet', 
        'engdwobdcd', 
        'engdwbdcd', 
        'engdwbll', 
        'engstafdcd', 
        'engstafll', 
        'engstafml', 
        'engsldcd', 
        'engsldcp', 
        'engcmssdcd', 
        'engcmssmp', 
        'urbrecptdc',
        'urbrecptwt', 
        'forpehrtdc',
        'hydclprs', 
        'awmmfpwwta'
    ]

    target = [
        'Rot1CropGr',
        'Rot1CropTy',
        'Rot2CropGr',
        'Rot2CropTy',
        'RotCropTyp',
        'CropGroup',
        'CropType',
    ]

    TEST_SHEDS = ["Lower Yakima", "Palouse", "Lower Crab"]
    catranges = {}

    maps = {
        "Very frequent": 1,
        "Frequent": .75,
        "Occasional": .5,
        "Rare": .25,
        "None": 0,
        "a": 0,
        "Excessively drained": 1,
        "Somewhat excessively drained": .85,
        "Well drained": .7,
        "Moderately well drained": .56,
        "Somewhat poorly drained": .42,
        "Poorly drained": .28,
        "Very poorly drained": .14, 
        "Good": 1,
        "Fair": .66,
        "Poor": .33, 
        "Not limited": 1,
        "Somewhat limited": .66,
        "Very limited": .33,
        "Not rated": 0,
        "Slight": 1,
        "Moderate": .66,
        "Severe": .33,
    }

    def removeleading(x):
        if x == 0:
            return 0
        if x[0] in "0123456789":
            if x[1:] == "":
                return 0
            return x[1:]
        return x

    for index, i in enumerate(years):
        df = pd.read_csv(main_dir + i).astype({'OBJECTID': 'int32'})
        df.fillna(0, inplace=True)

        for ind, x in df.iterrows():
            if x.watershed in TEST_SHEDS:
                ids.add(x.OBJECTID)
        for topic in cats:
            if topic in ['CropGroup', 'CropType', 'watershed']:
                continue
            for x in df[topic]:
                if type(x) == str and x != "a": 
                    if topic not in catranges:
                        catranges[topic] = {removeleading(x)}
                    else:
                        catranges[topic].add(removeleading(x))
        dfs.append(df)

    skip=['watersheds']

    for x in catranges:
        print(x, catranges[x])

    xar, yar, loc = [], [], []

    for i in dfs:
        for x in catranges:
            i[x] = i[x].map(removeleading)
            i.replace({x: maps}, inplace=True)

        temp = []
        for x in target:
            if x in i:
                for ind, item in enumerate(i[x]):
                    if len(temp) <= ind: 
                        if item in {"fallow", "Fallow"}:
                            temp.append(1)
                        else:
                            temp.append(0)
                    else:
                        if item in {"fallow", "Fallow"}:
                            temp[ind] = 1
        temploc = []
        yar.append(temp)
        print(i[[x for x in cats if x not in ['watershed', 'lat', 'lon'] + target]].shape)
        temploc.append(i[['lat', 'lon', 'watershed']])
        xar.append(i[[x for x in cats if x not in ['watershed', 'lat', 'lon'] + target]]) 
        _ = [loc.append(x) for x in temploc]


    return np.array([np.array(x) for x in xar], dtype=np.float32).reshape((-1, 30)), np.array([np.array(x) for x in yar], dtype=np.float32).flatten(), loc

    #farms = {x:[] for x in ids}

load_data()

"""
for i in dfs:
    for x in ids:
        a = i[i['OBJECTID'] == x]
        if not a.empty:
            farms[x].append(a.to_dict())
        else:
            farms[x].append(None)
    print("DoNE")
    break 

['CropGroup', 'CropType', 'TotalAcres', 'Shape_Leng', 'lat', 'long', 'Watershed', 'Shape_Le_1', 'Shape_Area', 'slopegradw', 'wtdepannmi', 'wtdepaprju', 'flodfreqdc', 'flodfreqma', 'pondfreqpr', 'aws025wta', 'aws0100wta', 'aws0150wta', 'drclassdcd', 'drclasswet', 'engdwodbcd', 'engdwdbcd', 'engdwbll', 'engstafdcd', 'engstafll', 'engstafml', 'engsldcd', 'engsldcp', 'engcmssdcd', 'engcmssmp', 'urbrecptdc', 'urbrecptwt', 'forpehrtdc','hydclprs', 'awmmfpwwta']


print(farms)

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(farms, f, ensure_ascii=False, indent=4)

#print(sorted(list(ids)))
"""