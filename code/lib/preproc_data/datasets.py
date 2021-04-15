#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data reading / handling for dys / xer datasets"""

__author__      = "Tuur Leeuwenberg"
__email__ = "A.M.Leeuwenberg-15@umcutrecht.nl"

import pandas as pd
import numpy as np
from lib.SPSSReader import read_sav
from copy import copy
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os
import base64
import hashlib
from cryptography.fernet import Fernet
# import pyreadr
# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
# pandas2ri.activate()


def transitive_closure(list_of_pairs):
    old = list_of_pairs
    closure = copy(old)
    for a1, a2 in old:
        for a3, a4 in old:
            if a2 == a3 and a1 != a4:
                closure.append((a1, a4))
    return list(set([tuple(x) for x in closure]))

def find_X_pairs(pairs, X_keys):
    #print(pairs)
    pairs = transitive_closure(pairs)
    #print(pairs)
    #exit()
    constraints = set([])
    locations = {k:i for i,k in enumerate(X_keys)}
    dummy_token = '<DUMMY>'
    for o1, o2 in pairs:
        o1_matches, o2_matches = set([k.replace(o1,dummy_token) for k in X_keys if o1 in k]), set([k.replace(o1,dummy_token) for k in X_keys if o1 in k])
        for match in o1_matches.intersection(o2_matches):
            constraints.add((match.replace(dummy_token, o1), match.replace(dummy_token,o2)))
    return constraints, [(locations[m1], locations[m2]) for (m1,m2) in constraints if m1 in locations and m2 in locations]

def describe_data(df, dir, y_name, pairplot=0):
    if not os.path.exists(dir):
        os.makedirs(dir)
    df.describe(include='all').to_csv(dir + '/descriptions.csv')
    int_keys = []
    for key in df.keys():
        try:
            intdf = df[key].astype(int)
        except:
            continue
        if np.array_equal(df[key], intdf):
            int_keys.append(key)
    df[int_keys].apply(pd.Series.value_counts).to_csv(dir + '/value_counts.csv')
    #other_keys = [k for k in df.keys() if not k in int_keys]

    if pairplot:
        sns.set(style="ticks")
        pp = sns.pairplot(df.dropna(), hue=y_name,plot_kws={'alpha': 0.25})
        pp.savefig(dir + "/scatter_matrix.png")
        #scatter_matrix(, alpha=0.2, figsize=(3*len(xdf.keys())+len(ydf.keys()), 3*len(xdf.keys())+len(ydf.keys())), diagonal='kde')
        #plt.savefig(dir + "/scatter_matrix.png")

# Source https://nitratine.net/blog/post/encryption-and-decryption-in-python/
def save_python_object_encrypted(obj, file_path, password):
    key = base64.b64encode(hashlib.md5(password.encode()).hexdigest().encode())
    byte_obj = pickle.dumps(obj)
    crypt = Fernet(key)
    encoded = crypt.encrypt(byte_obj)
    with open(file_path, 'wb') as f:
        f.write(encoded)

def load_python_object_encrypted(file_path, password):
    key = base64.b64encode(hashlib.md5(password.encode()).hexdigest().encode())
    crypt = Fernet(key)
    with open(file_path, 'rb') as f:
        encoded = f.read()
    byte_obj = crypt.decrypt(encoded)
    obj = pickle.loads(byte_obj)
    return obj

# print('---')
# test_obj = {a:str(a+1) for a in range(10)}
# print(test_obj)
# save_python_object_encrypted(test_obj, 'test_file.encr','1234')
# dec_test_obj = load_python_object_encrypted('test_file.encr','1234')
# print(dec_test_obj)
# exit()

class OARs:

    PARi='PARi'
    PARc='PARc'
    SUBMi='SUBMi'
    SUBMc='SUBMc'
    CRICO='CRICO'
    EIM='EIM'
    GLOT='GLOT'
    SUPGL='SUPGL'
    PCMsup='PCMsup'
    PCMmed='PCMmed'
    PCMinf='PCMinf'
    ORAL_ATL='ORAL_ATL'

    ALL=[PARi,PARc,SUBMi,SUBMc,CRICO,EIM,SUPGL,PCMsup,PCMmed,PCMinf,ORAL_ATL]

    def extend(string):
        return [oar for oar in OARs.ALL if string in oar]

    def extend_list_of_pairs(list_of_pairs):
        extended = []
        for a1, a2 in list_of_pairs:
            for e1 in OARs.extend(a1):
                for e2 in OARs.extend(a2):
                    extended.append((e1,e2))
        return extended

    def transitive_closure(list_of_pairs):
        extended_list_of_pairs = OARs.extend_list_of_pairs(list_of_pairs)
        old = set(extended_list_of_pairs)
        closure = copy(old)
        for a1, a2 in extended_list_of_pairs:
            for a3, a4 in extended_list_of_pairs:
                if a2 == a3 and a1 != a4:
                    closure.add((a1, a4))
        return list(closure)



class XerostomiaDataset(object):
    sav_path = "/media/sf_HTx/C_Data/4 Final_data/1 Unimputed/2017-07-18  DB External validation patient-rated XEROSTOMIA_THINC.sav"
    oar_order = OARs.transitive_closure([('SUBM','PAR')]) # PAR > SUBM

    def __init__(self, oars=[OARs.PARi, OARs.PARc, OARs.SUBMi, OARs.SUBMc], dmeans=True, dvolume=False, hospitals=['VUMC','UMCG'], techniques=['3DCRT','IMRT','VMAT']):
        self.oars = oars
        self.dmeans=dmeans
        self.dvolume=dvolume
        self.hospitals=hospitals # can be VUMC or UMCG
        self.techniques=techniques # can be IMRT or 3DCRT or VMAT
        self.X, self.y, self.sdf = self.get_data()

    def get_data(self):
        df, _ = read_sav(self.sav_path)
        hospital_indices = {'VUMC':[0],'UMCG':[1,3]}
        technique_indices = {'3DCRT':[0],'IMRT':[1],'VMAT':[2]}

        # Filter on selected hospitals
        df = df[df['Beetz'].isin([v for l in [hospital_indices[hospital] for hospital in self.hospitals] for v in l])]
        # Filter on technique
        df = df[df['RT_TECH'].isin([v for l in [technique_indices[technique] for technique in self.techniques] for v in l])]

        # select patients with no baseline complications grade 2 or higher
        df = df[df['XER_BASELINE_corrected'].isin([0,1])]

        df = df[df['RESP_prXER_M06'].isin([1.0,0.0])]  # No NaN outcome data
        df = df.sort_values(by='STARTvolgorde')

        df = df[df['LOCTUM'].isin([1.0, 2.0, 3.0, 4.0, 5.0])]

        # y: (Xerostomia 6M Gr2+)
        y_df = pd.DataFrame()
        y_df['y'] = df['RESP_prXER_M06']

        X_df = pd.get_dummies(df['XER_BASELINE_corrected_3cat'], prefix='BaselineCompl', drop_first=True)

        X_Tumorloc_tmp = pd.get_dummies(df['LOCTUM'], prefix='TLOC')
        X_Tumorloc = pd.DataFrame()
        X_Tumorloc['TLOC_oral_cavity'] = X_Tumorloc_tmp['TLOC_1.0']
        X_Tumorloc['TLOC_oropharynx'] = X_Tumorloc_tmp['TLOC_2.0']
        X_Tumorloc['TLOC_nasopharynx'] = X_Tumorloc_tmp['TLOC_3.0']
        X_Tumorloc['TLOC_hypopharynx'] = X_Tumorloc_tmp['TLOC_4.0']
        X_Tumorloc['TLOC_larynx'] = X_Tumorloc_tmp['TLOC_5.0']
        X_df = pd.concat([X_df,X_Tumorloc], axis=1)

        if self.dmeans:
            X_oars_dmeans = df.filter(regex="|".join(['^'+soar+"_Dmean_REAL" for soar in self.oars]))
            X_df = pd.concat([X_df, X_oars_dmeans], axis=1)

        if self.dvolume:
            X_oars_dvolume = df.filter(regex="|".join(['^'+oar+'.*_v.*_REAL' for oar in self.oars]))
            X_df = pd.concat([X_df, X_oars_dvolume], axis=1)

        return X_df, y_df, df



class Beetz2012Xer(XerostomiaDataset):

    def get_data(self, development_data=False): # returns the data NOT used for model development
        df, _ = read_sav(self.sav_path)
        df = df[df['RESP_prXER_M06'].isin([1.0,0.0])]  # No NaN data


        df= df[df['Beetz'].isin([0,1])] if development_data else df[df['Beetz'].isin([3])]
        df = df[df['XER_BASELINE_corrected'].isin([0,1])]
        df = df.sort_values(by='STARTvolgorde')

        x1 = df['PARc_Dmean_REAL']  # Dm_PARc
        x2 = df['XER_BASELINE_corrected'] # Baseline_XER 0 = None, 1 = Minor
        X_df = pd.concat([x1, x2], axis=1)

        # y: (Xerostomia 6M Gr2+)
        y_df = pd.DataFrame()
        y_df['y'] = df['RESP_prXER_M06']
        return X_df, y_df, df

    def get_model(self):
        model = LogisticRegression(penalty='none',solver='saga', max_iter=1000)
        model.coef_ = np.array([[0.047, 0.720]])
        model.intercept_ = np.array([- 1.443])
        return model

    def get_dev_size(self):
        return 161



class DysphagiaDataset(object):
    sav_path = sav_path = "/media/sf_HTx/C_Data/4 Final_data/1 Unimputed/2017-06-09  DB DYSFAGIE RTOG opgeschoond COMPLETE DATA THINC.sav"
    oar_order = OARs.transitive_closure([('SUBM','PAR'),('PCM','PAR'),('ORAL','PCM')])

    def __init__(self, oars=[OARs.PCMsup, OARs.PCMmed, OARs.PCMinf, OARs.ORAL_ATL], dmeans=True, dvolume=False, hospitals=['VUMC','UMCG'], techniques=['3DCRT','IMRT']):
        self.oars = oars
        self.dmeans=dmeans
        self.dvolume=dvolume
        self.hospitals=hospitals # can be VUMC or UMCG
        self.techniques=techniques # can be IMRT or 3DCRT or VMAT
        self.X, self.y, self.sdf = self.get_data()

    def X(self):
        return self.X

    def y(self):
        return self.y

    def get_data(self):
        df, _ = read_sav(self.sav_path)
        df = df.sort_values(by='ORDER')

        hospital_indices = {'VUMC':[1],'UMCG':[2]}
        technique_indices = {'3DCRT':[0],'IMRT':[1]}

        # Filter on selected hospitals
        df = df[df['HOSP'].isin([v for l in [hospital_indices[hospital] for hospital in self.hospitals] for v in l])]
        # Filter on technique
        df = df[df['RT_TEC_2cat'].isin([v for l in [technique_indices[technique] for technique in self.techniques] for v in l])]


        #df = df[df['HOSP'] == 2]  # UMCG

        df = df[df['LOCTUM_cat'].isin([1.0, 2.0, 3.0, 4.0, 5.0])]

        # select patients with no baseline complications grade 2 or higher
        df = df[df['DYSFAGIE_UMCGshort_W0'].isin([0])]


        # y: (Dysphagia 6M Gr2+)
        y_df = pd.DataFrame()
        y_df['y'] = df['RESP_DYSFctcae_M06']

        X_Baseline = pd.get_dummies(df['DYSFAGIE_UMCGshort_W0'], prefix='BaselineCompl', drop_first=True)

        X_Tumorloc_tmp = pd.get_dummies(df['LOCTUM_cat'], prefix='TLOC')
        X_Tumorloc = pd.DataFrame()
        X_Tumorloc['TLOC_oral_cavity'] = X_Tumorloc_tmp['TLOC_1.0']
        X_Tumorloc['TLOC_oropharynx'] = X_Tumorloc_tmp['TLOC_2.0']
        X_Tumorloc['TLOC_nasopharynx'] = X_Tumorloc_tmp['TLOC_3.0']
        X_Tumorloc['TLOC_hypopharynx'] = X_Tumorloc_tmp['TLOC_4.0']
        X_Tumorloc['TLOC_larynx'] = X_Tumorloc_tmp['TLOC_5.0']

        X_df = pd.concat([X_Baseline,X_Tumorloc], axis=1)

        if self.dmeans:
            X_oars_dmeans = df.filter(regex="|".join(['^'+soar+"_Dmean_REAL" for soar in self.oars]))
            X_df = pd.concat([X_df, X_oars_dmeans], axis=1)

        if self.dvolume:
            X_oars_dvolume = df.filter(regex="|".join(['^'+oar+'.*_v.*_REAL' for oar in self.oars]))
            X_df = pd.concat([X_df, X_oars_dvolume], axis=1)

        return X_df, y_df, df

class Christianen2012Dys(DysphagiaDataset):

    def get_data(self, development_data=False):
        df, _ = read_sav(self.sav_path)
        df = df.sort_values(by='ORDER')
        df = df[df['Christianen_studie'] == 0]  # validation set / returns the data NOT used for model development
        df = df[df['RT_TEC_2cat']==1] # IMRT only
        df = df[df['BASELINE_DYSF']==0] # only patients that have no moderate-to-sever DYS at baseline

        x1 = df['PCMsup_Dmean_REAL']  # PCMsup_Dmean_REAL
        x2 = df['SUPGL_Dmean_REAL'] # SUPGL_Dmean_REAL
        X_df = pd.concat([x1, x2], axis=1)

        # y: (Dysphagia 6M Gr2+)
        y_df = pd.DataFrame()
        y_df['y'] = df['RESP_DYSFctcae_M06']

        return X_df, y_df, df

    def get_model(self):
        model = LogisticRegression(penalty='none',solver='saga', max_iter=1000)
        model.coef_ = np.array([[0.057, 0.037]])
        model.intercept_ = np.array([- 6.09])
        return model

    def get_dev_size(self):
        return 353


