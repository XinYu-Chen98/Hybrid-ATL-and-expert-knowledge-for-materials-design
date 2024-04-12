#!/usr/bin/env python
# coding: utf-8

# In[121]:


#import matminer featurizers and pymatgen
from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matminer.datasets import load_dataset
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital, IonProperty
from matminer.featurizers.structure import (SiteStatsFingerprint, StructuralHeterogeneity,
                                            ChemicalOrdering, StructureComposition, MaximumPackingEfficiency)
from matminer.featurizers.conversions import DictToObject
#import torch
import torch
import torch.nn as nn
#import ASE/mendeleev
from mendeleev import element
from ase.io import read
from ase.build import molecule
from ase import Atoms
#import basics
from os import listdir   
import os 
import pandas as pd
import csv
import numpy as np
import eventlet
import random
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm


# In[131]:


#functions and dicts
def build_feature_from_cif(path_cif):
    featurizer = MultipleFeaturizer([
    #SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"),
    #StructuralHeterogeneity(),
    #ChemicalOrdering(),
    #MaximumPackingEfficiency(),
    #SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017"),
    StructureComposition(Stoichiometry()),
    StructureComposition(ElementProperty.from_preset("magpie")),
    StructureComposition(ValenceOrbital(props=['frac'])),
    StructureComposition(IonProperty(fast=True))
    ])
    #help(featurizer)
    NAME = []
    STR = []
    ciflist = listdir(path_cif)
    for i in tqdm(range(len(ciflist))):#read POSCAR or cif from file path
        try:
            cif = ciflist[i]
            structuremg = Structure.from_file(path_cif+cif)
            STR.append(structuremg)
            NAME.append(cif)
        except:
            print('Error! When featurize'+cif)
    X = featurizer.featurize_many(STR,ignore_errors=True)
    return(STR,NAME,X)

dict_Rnum = {'1': 1, '2': 1, '3': 1,'4': 1, '5': 1, '6': 1,'7': 1, '8': 1, '9': 1,'10': 1, 
             '11': 1, '12': 1, '13': 1,'14': 1, '15': 1, '16': 2,'17': 2, '18': 2, '19': 2,'20': 2, 
             '21': 2, '22': 2, '23': 2,'24': 2, '25': 2, '26': 2,'27': 2, '28': 2, '29': 2,'30': 2,
             '31': 2, '32': 2, '33': 2,'34': 2, '35': 2, '36': 2,'37': 2, '38': 2, '39': 2,'40': 2,
             '41': 2, '42': 2, '43': 2,'44': 2, '45': 2, '46': 2,'47': 2, '48': 2, '49': 2,'50': 2,
             '51': 2, '52': 2, '53': 2,'54': 2, '55': 2, '56': 2,'57': 2, '58': 2, '59': 2,'60': 2,
             '61': 2, '62': 2, '63': 2,'64': 2, '65': 2, '66': 2,'67': 2, '68': 2, '69': 2,'70': 2,
             '71': 2, '72': 2, '73': 2,'74': 2, '75': 4, '76': 4,'77': 4, '78': 4, '79': 4,'80': 4, 
             '81': 4, '82': 4, '83': 6,'84': 6, '85': 6, '86': 6,'87': 6, '88': 6, '89': 4,'90': 4, 
             '91': 4, '92': 4, '93': 4,'94': 4, '95': 4, '96': 4,'97': 4, '98': 4, '99': 4,'100': 4,
             '101': 4, '102': 4, '103': 4,'104': 4, '105': 4, '106': 4,'107': 4, '108': 4, '109': 4,'110': 4, 
             '111': 4, '112': 4, '113': 4,'114': 4, '115': 4, '116': 4,'117': 4, '118': 4, '119': 4,'120': 4, 
             '121': 4, '122': 4, '123': 6,'124': 6, '125': 6, '126': 6,'127': 6, '128': 6, '129': 6,'130': 6,
             '131': 6, '132': 6, '133': 6,'134': 6, '135': 6, '136': 6,'137': 6, '138': 6, '139': 6,'140': 6,
             '141': 6, '142': 6, '143': 3,'144': 3, '145': 3, '146': 3,'147': 5, '148': 5, '149': 3,'150': 3,
             '151': 3, '152': 3, '153': 3,'154': 3, '155': 3, '156': 3,'157': 3, '158': 3, '159': 3,'160': 3,
             '161': 3, '162': 5, '163': 5,'164': 5, '165': 5, '166': 5,'167': 5, '168': 6, '169': 6,'170': 6,
             '171': 6, '172': 6, '173': 6,'174': 5, '175': 10, '176': 10,'177': 6, '178': 6, '179': 6,'180': 6, 
             '181': 6, '182': 6, '183': 6,'184': 6, '185': 6, '186': 6,'187': 5, '188': 5, '189': 5,'190': 5, 
             '191': 10, '192': 10, '193': 10,'194': 10, '195': 2, '196': 2,'197': 2, '198': 2, '199': 2,'200': 2,
             '201': 2, '202': 2, '203': 2,'204': 2, '205': 2, '206': 2,'207': 4, '208': 4, '209': 4,'210': 4, 
             '211': 4, '212': 4, '213': 4,'214': 4, '215': 4, '216': 4,'217': 4, '218': 4, '219': 4,'220': 4, 
             '221': 6, '222': 6, '223': 6,'224': 6, '225': 6, '226': 6,'227': 6, '228': 6, '229': 6,'230': 6,}
dict_Mnum = {'1': 0, '2': 0, '3': 0,'4': 0, '5': 0, '6': 1,'7': 1, '8': 1, '9': 1,'10': 1, 
             '11': 1, '12': 1, '13': 1,'14': 1, '15': 1, '16': 0,'17': 0, '18': 0, '19': 0,'20': 0, 
             '21': 0, '22': 0, '23': 0,'24': 0, '25': 2, '26': 2,'27': 2, '28': 2, '29': 2,'30': 2,
             '31': 2, '32': 2, '33': 2,'34': 2, '35': 2, '36': 2,'37': 2, '38': 2, '39': 2,'40': 2,
             '41': 2, '42': 2, '43': 2,'44': 2, '45': 2, '46': 2,'47': 2, '48': 2, '49': 2,'50': 2,
             '51': 2, '52': 2, '53': 2,'54': 2, '55': 2, '56': 2,'57': 2, '58': 2, '59': 2,'60': 2,
             '61': 2, '62': 2, '63': 2,'64': 2, '65': 2, '66': 2,'67': 2, '68': 2, '69': 2,'70': 2,
             '71': 2, '72': 2, '73': 2,'74': 2, '75': 0, '76': 0,'77': 0, '78': 0, '79': 0,'80': 0, 
             '81': 0, '82': 0, '83': 0,'84': 0, '85': 0, '86': 0,'87': 0, '88': 0, '89': 0,'90': 0, 
             '91': 0, '92': 0, '93': 0,'94': 0, '95': 0, '96': 0,'97': 0, '98': 0, '99': 0,'100': 4,
             '101': 4, '102': 4, '103': 4,'104': 4, '105': 4, '106': 4,'107': 4, '108': 4, '109': 4,'110': 4, 
             '111': 2, '112': 2, '113': 2,'114': 2, '115': 2, '116': 2,'117': 2, '118': 2, '119': 2,'120': 2, 
             '121': 2, '122': 2, '123': 4,'124': 4, '125': 4, '126': 4,'127': 4, '128': 4, '129': 4,'130': 4,
             '131': 4, '132': 4, '133': 4,'134': 4, '135': 4, '136': 4,'137': 4, '138': 4, '139': 4,'140': 4,
             '141': 4, '142': 4, '143': 0,'144': 0, '145': 0, '146': 0,'147': 0, '148': 0, '149': 0,'150': 0,
             '151': 0, '152': 0, '153': 0,'154': 0, '155': 0, '156': 3,'157': 3, '158': 3, '159': 3,'160': 3,
             '161': 3, '162': 3, '163': 3,'164': 3, '165': 3, '166': 3,'167': 3, '168': 0, '169': 0,'170': 0,
             '171': 0, '172': 0, '173': 0,'174': 0, '175': 0, '176': 0,'177': 0, '178': 0, '179': 0,'180': 0, 
             '181': 0, '182': 0, '183': 6,'184': 6, '185': 6, '186': 6,'187': 3, '188': 3, '189': 3,'190': 3, 
             '191': 6, '192': 6, '193': 6,'194': 6, '195': 0, '196': 0,'197': 0, '198': 0, '199': 0,'200': 2,
             '201': 2, '202': 2, '203': 2,'204': 2, '205': 2, '206': 2,'207': 0, '208': 0, '209': 0,'210': 0, 
             '211': 0, '212': 0, '213': 0,'214': 0, '215': 2, '216': 2,'217': 2, '218': 2, '219': 2,'220': 2, 
             '221': 4, '222': 4, '223': 4,'224': 4, '225': 4, '226': 4,'227': 4, '228': 4, '229': 4,'230': 4,}
dict_HM =    {'1': 0, '2': 0, '3': 0,'4': 0, '5': 0, '6': 0,'7': 0, '8': 0, '9': 0,'10': 0, 
             '11': 0, '12': 0, '13': 0,'14': 0, '15': 0, '16': 0,'17': 0, '18': 0, '19': 0,'20': 0, 
             '21': 0, '22': 0, '23': 0,'24': 0, '25': 0, '26': 0,'27': 0, '28': 0, '29': 0,'30': 0,
             '31': 0, '32': 0, '33': 0,'34': 0, '35': 0, '36': 0,'37': 0, '38': 0, '39': 0,'40': 0,
             '41': 0, '42': 0, '43': 0,'44': 0, '45': 0, '46': 0,'47': 1, '48': 1, '49': 1,'50': 1,
             '51': 1, '52': 1, '53': 1,'54': 1, '55': 1, '56': 1,'57': 1, '58': 1, '59': 1,'60': 1,
             '61': 1, '62': 1, '63': 1,'64': 1, '65': 1, '66': 1,'67': 1, '68': 1, '69': 1,'70': 1,
             '71': 1, '72': 1, '73': 1,'74': 1, '75': 0, '76': 0,'77': 0, '78': 0, '79': 0,'80': 0, 
             '81': 0, '82': 0, '83': 1,'84': 1, '85': 1, '86': 1,'87': 1, '88': 1, '89': 0,'90': 0, 
             '91': 0, '92': 0, '93': 0,'94': 0, '95': 0, '96': 0,'97': 0, '98': 0, '99': 0,'100': 0,
            '101': 0, '102': 0, '103': 0,'104': 0, '105': 0, '106': 0,'107': 0, '108': 0, '109': 0,'110': 0, 
            '111': 0, '112': 0, '113': 0,'114': 0, '115': 0, '116': 0,'117': 0, '118': 0, '119': 0,'120': 0, 
            '121': 0, '122': 0, '123': 1,'124': 1, '125': 1, '126': 1,'127': 1, '128': 1, '129': 1,'130': 1,
            '131': 1, '132': 1, '133': 1,'134': 1, '135': 1, '136': 1,'137': 1, '138': 1, '139': 1,'140': 1,
            '141': 1, '142': 1, '143': 0,'144': 0, '145': 0, '146': 0,'147': 0, '148': 0, '149': 0,'150': 0,
            '151': 0, '152': 0, '153': 0,'154': 0, '155': 0, '156': 0,'157': 0, '158': 0, '159': 0,'160': 0,
            '161': 0, '162': 0, '163': 0,'164': 0, '165': 0, '166': 0,'167': 0, '168': 0, '169': 0,'170': 0,
            '171': 0, '172': 0, '173': 0,'174': 1, '175': 1, '176': 1,'177': 0, '178': 0, '179': 0,'180': 0, 
            '181': 0, '182': 0, '183': 0,'184': 0, '185': 0, '186': 0,'187': 1, '188': 1, '189': 1,'190': 1, 
            '191': 1, '192': 1, '193': 1,'194': 1, '195': 0, '196': 0,'197': 0, '198': 0, '199': 0,'200': 1,
            '201': 1, '202': 1, '203': 1,'204': 1, '205': 1, '206': 1,'207': 0, '208': 0, '209': 0,'210': 0, 
            '211': 0, '212': 0, '213': 0,'214': 0, '215': 0, '216': 0,'217': 0, '218': 0, '219': 0,'220': 0, 
            '221': 1, '222': 1, '223': 1,'224': 1, '225': 1, '226': 1,'227': 1, '228': 1, '229': 1,'230': 1,}

def get_atom_layers_thickness(structure):
    position = structure.cart_coords
    cell = structure.lattice.matrix
    c = cell[2].tolist()[2]
    z_cord = []
    for i in range(len(position)):
        z = round(position[i][2],2)
        if z not in z_cord:
            z_cord.append(z)
    atom_layers = len(z_cord)
    thickness = max(z_cord)-min(z_cord)
    if thickness/c > 0.7:
        zdown,zup = [],[]
        for j in range(len(z_cord)):
            if z_cord[j]/c < 0.5:
                zdown.append(z_cord[j])
            else:
                zup.append(z_cord[j])
        thickness = c-min(zup)+max(zdown)
    return(atom_layers,thickness)

def get_space_group_number(structure):
    analyzer = SpacegroupAnalyzer(structure)
    spacegroup_number = analyzer.get_space_group_number()
    return(str(spacegroup_number))

def get_en(a):
    en = []
    en_s,en_p,en_d,en_f = 0,0,0,0
    for i in range(len(a)-1):
        if a[i] == 's':
            en_s += int(a[i+1])
        if a[i] == 'p':
            en_p += int(a[i+1])
        if a[i] == 'd':
            en_d += int(a[i+1])
        if a[i] == 'f':
            en_f += int(a[i+1])
    en.append(en_s)
    en.append(en_p)
    en.append(en_d)
    en.append(en_f)
    return(en)

def make_cbfv(ele_number):
    cbfv = []
    b = element(ele_number[0])
    atomnum = []#原子序号
    atomnumsum = 0
    for i in range(len(b)):
        atomnum.append(b[i].atomic_number)
        atomnumsum += b[i].atomic_number*int(ele_number[1][i])
    cbfv.append(atomnumsum/len(atomnum))
    cbfv.append(max(atomnum))
    cbfv.append(min(atomnum))
    cbfv.append(np.std(atomnum))
    atomwt = []#原子质量
    atomwtsum = 0
    for i in range(len(b)):
        atomwt.append(b[i].atomic_weight)
        atomwtsum += b[i].atomic_weight*int(ele_number[1][i])
    cbfv.append(atomwtsum/len(atomwt))
    cbfv.append(max(atomwt))
    cbfv.append(min(atomwt))
    cbfv.append(np.std(atomwt))
    atompr = []#行数
    atomprsum = 0
    for i in range(len(b)):
        atompr.append(b[i].period)
        atomprsum += b[i].period*int(ele_number[1][i])
    cbfv.append(atomprsum/len(atompr))
    cbfv.append(max(atompr))
    cbfv.append(min(atompr))
    cbfv.append(np.std(atompr))
    atomgr = []#列数
    atomgrsum = 0
    for i in range(len(b)):
        temp = b[i].group_id
        if temp != None:
            atomgr.append(b[i].group_id)
            atomgrsum += b[i].group_id*int(ele_number[1][i])
    cbfv.append(atomgrsum/len(atomgr))
    cbfv.append(max(atomgr))
    cbfv.append(min(atomgr))
    cbfv.append(np.std(atomgr))
    atommn = []#门捷列夫数
    atommnsum = 0
    for i in range(len(b)):
        atommn.append(b[i].mendeleev_number)
        atommnsum += b[i].mendeleev_number*int(ele_number[1][i])
    cbfv.append(atommnsum/len(atommn))
    cbfv.append(max(atommn))
    cbfv.append(min(atommn))
    cbfv.append(np.std(atommn))
    atomar = []#原子半径
    atomarsum = 0
    for i in range(len(b)):
        temp = b[i].atomic_radius
        if temp != None:
            atomar.append(temp)
            atomarsum += temp*int(ele_number[1][i])
    cbfv.append(atomarsum/len(atomar))
    cbfv.append(max(atomar))
    cbfv.append(min(atomar))
    cbfv.append(np.std(atomar))
    atomcr = []#共价半径
    atomcrsum = 0
    for i in range(len(b)):
        atomcr.append(b[i].covalent_radius)
        atomcrsum += b[i].covalent_radius*int(ele_number[1][i])
    cbfv.append(atomcrsum/len(atomcr))
    cbfv.append(max(atomcr))
    cbfv.append(min(atomcr))
    cbfv.append(np.std(atomcr))
    atompen = []#Pauli en
    atompensum = 0
    for i in range(len(b)):
        temp = b[i].en_pauling
        if temp != None:
            atompen.append(b[i].en_pauling)
            atompensum += b[i].en_pauling*int(ele_number[1][i])
    cbfv.append(atompensum/len(atompen))
    cbfv.append(max(atompen))
    cbfv.append(min(atompen))
    cbfv.append(np.std(atompen))
    atomaen = []#Allen en
    atomaensum = 0
    for i in range(len(b)):
        temp = b[i].en_allen
        if temp != None:
            atomaen.append(b[i].en_allen)
            atomaensum += b[i].en_allen*int(ele_number[1][i])
    cbfv.append(atomaensum/len(atomaen))
    cbfv.append(max(atomaen))
    cbfv.append(min(atomaen))
    cbfv.append(np.std(atomaen))
    atomgen = []#Ghosh en
    atomgensum = 0
    for i in range(len(b)):
        temp = b[i].en_ghosh
        if temp != None:
            atomgen.append(b[i].en_ghosh)
            atomgensum += b[i].en_ghosh*int(ele_number[1][i])
    cbfv.append(atomgensum/len(atomgen))
    cbfv.append(max(atomgen))
    cbfv.append(min(atomgen))
    cbfv.append(np.std(atomgen))
    atommr = []#metalic radii
    atommrsum = 0
    for i in range(len(b)):
        temp = b[i].metallic_radius
        if temp != None:
            atommr.append(temp)
            atommrsum += temp*int(ele_number[1][i])
    if atommr == []:
        cbfv.append(0)
        cbfv.append(0)
        cbfv.append(0)
        cbfv.append(0)
    else:
        cbfv.append(atommrsum/len(atommr))
        cbfv.append(max(atommr))
        cbfv.append(min(atommr))
        cbfv.append(np.std(atommr))
    atomsen = []#s electron num
    atomsensum = 0
    for i in range(len(b)):
        temp = get_en(str(b[i].ec))[0]
        atomsen.append(temp)
        atomsensum += temp*int(ele_number[1][i])
    cbfv.append(atomsensum/len(atomsen))
    cbfv.append(max(atomsen))
    cbfv.append(min(atomsen))
    cbfv.append(np.std(atomsen))
    atompen = []#p electron num
    atompensum = 0
    for i in range(len(b)):
        temp = get_en(str(b[i].ec))[1]
        atompen.append(temp)
        atompensum += temp*int(ele_number[1][i])
    cbfv.append(atompensum/len(atompen))
    cbfv.append(max(atompen))
    cbfv.append(min(atompen))
    cbfv.append(np.std(atompen))
    atomden = []#d electron num
    atomdensum = 0
    for i in range(len(b)):
        temp = get_en(str(b[i].ec))[2]
        atomden.append(temp)
        atomdensum += temp*int(ele_number[1][i])
    cbfv.append(atomdensum/len(atomden))
    cbfv.append(max(atomden))
    cbfv.append(min(atomden))
    cbfv.append(np.std(atomden))
    atomfie = []#first ionized energy
    atomfiesum = 0
    for i in range(len(b)):
        temp = str(b[i].ionenergies)
        temp = float(temp[4:9])
        atomfie.append(temp)
        atomfiesum += temp*int(ele_number[1][i])
    cbfv.append(atomfiesum/len(atomfie))
    cbfv.append(max(atomfie))
    cbfv.append(min(atomfie))
    cbfv.append(np.std(atomfie))
    atompo = []#polarizaion
    atomposum = 0
    for i in range(len(b)):
        temp = b[i].dipole_polarizability
        atompo.append(temp)
        atomposum += temp*int(ele_number[1][i])
    cbfv.append(atomposum/len(atompo))
    cbfv.append(max(atompo))
    cbfv.append(min(atompo))
    cbfv.append(np.std(atompo))
    atommp = []#melting point
    atommpsum = 0
    for i in range(len(b)):
        temp = 0
        atommp.append(temp)
        atommpsum += temp*int(ele_number[1][i])
    cbfv.append(atommpsum/len(atommp))
    cbfv.append(max(atommp))
    cbfv.append(min(atommp))
    cbfv.append(np.std(atommp))
    atombp = []#boling point
    atombpsum = 0
    for i in range(len(b)):
        temp = 0
        atombp.append(temp)
        atombpsum += temp*int(ele_number[1][i])
    cbfv.append(atombpsum/len(atombp))
    cbfv.append(max(atombp))
    cbfv.append(min(atombp))
    cbfv.append(np.std(atombp))
    atomds = []#density
    atomdssum = 0
    for i in range(len(b)):
        temp = 0
        atomds.append(temp)
        atomdssum += temp*int(ele_number[1][i])
    cbfv.append(atomdssum//len(atomds))
    cbfv.append(max(atomds))
    cbfv.append(min(atomds))
    cbfv.append(np.std(atomds))
    atomsh = []#specific heat
    atomshsum = 0
    for i in range(len(b)):
        temp = 0
        if temp != None:
            atomsh.append(temp)
            atomshsum += temp*int(ele_number[1][i])
    if atomsh == []:
        cbfv.append(0)
        cbfv.append(0)
        cbfv.append(0)
        cbfv.append(0)
    else:   
        cbfv.append(atomshsum/len(atomsh))
        cbfv.append(max(atomsh))
        cbfv.append(min(atomsh))
        cbfv.append(np.std(atomsh))
    atomvh = []#vaperization heat
    atomvhsum = 0
    for i in range(len(b)):
        temp = 0
        if temp != None:
            atomvh.append(temp)
            atomvhsum += temp*int(ele_number[1][i])
    if atomvh == []:
        cbfv.append(0)
        cbfv.append(0)
        cbfv.append(0)
        cbfv.append(0)
    else:
        cbfv.append(atomvhsum/len(atomvh))
        cbfv.append(max(atomvh))
        cbfv.append(min(atomvh))
        cbfv.append(np.std(atomvh))
    atomfh = []#fusion heat
    atomfhsum = 0
    for i in range(len(b)):
        temp = 0
        if temp != None:
            atomfh.append(temp)
            atomfhsum += temp*int(ele_number[1][i])
    if atomfh == []:
        cbfv.append(0)
        cbfv.append(0)
        cbfv.append(0)
        cbfv.append(0)
    else:    
        cbfv.append(atomfhsum/len(atomfh))
        cbfv.append(max(atomfh))
        cbfv.append(min(atomfh))
        cbfv.append(np.std(atomfh))
    atomtc = []#thermal_conductivity
    atomtcsum = 0
    for i in range(len(b)):
        temp = 0
        if temp != None:
            atomtc.append(temp)
            atomtcsum += temp*int(ele_number[1][i])
    if atomtc == []:
        cbfv.append(0)
        cbfv.append(0)
        cbfv.append(0)
        cbfv.append(0)
    else:
        cbfv.append(atomtcsum/len(atomtc))
        cbfv.append(max(atomtc))
        cbfv.append(min(atomtc))
        cbfv.append(np.std(atomtc))
    atomhf = []#heat_of_formation
    atomhfsum = 0
    for i in range(len(b)):
        temp = 0
        if temp != None:
            atomhf.append(temp)
            atomhfsum += temp*int(ele_number[1][i])
    if atomhf == []:
        cbfv.append(0)
        cbfv.append(0)
        cbfv.append(0)
        cbfv.append(0)
    else:
        cbfv.append(atomhfsum/len(atomhf))
        cbfv.append(max(atomhf))
        cbfv.append(min(atomhf))
        cbfv.append(np.std(atomhf))
    atomlc = []#llattice_constant
    atomlcsum = 0
    for i in range(len(b)):
        temp = 0
        if temp != None:
            atomlc.append(temp)
            atomlcsum += temp*int(ele_number[1][i])
    cbfv.append(atomlcsum/len(atomlc))
    cbfv.append(max(atomlc))
    cbfv.append(min(atomlc))
    cbfv.append(np.std(atomlc))
    return(cbfv)


# In[132]:


#generate MAGPIE feature set
path_cif = '/mnt/c/Users/azere/Desktop/C2DB2021/Test_POSCAR/'
struct,name,feature_magpie = build_feature_from_cif(path_cif)
#print(struct,name,feature)
print(feature_magpie)


# In[133]:


# ATL extracted features
class MLPmodel(nn.Module):
    def __init__(self,a,b,c,d,e,f,g,h,i,j):
        super(MLPmodel,self).__init__()
        self.hidden1=nn.Sequential(
            nn.Linear(145,a),
            #nn.BatchNorm1d(num_features=a),
            nn.ReLU(), 
        )
        self.hidden2=nn.Sequential(
            nn.Linear(a,b),
            #nn.BatchNorm1d(num_features=b),
            nn.ReLU(),
        )
        self.hidden3=nn.Sequential(
            nn.Linear(b,c),
            #nn.BatchNorm1d(num_features=c),
            nn.ReLU(),
        )
        self.hidden4=nn.Sequential(
            nn.Linear(c,d),
            #nn.BatchNorm1d(num_features=d),
            nn.ReLU(),
        )
        self.hidden5=nn.Sequential(
            nn.Linear(d,e),
            #nn.BatchNorm1d(num_features=e),
            nn.ReLU(),
        )
        self.hidden6=nn.Sequential(
            nn.Linear(e,f),
            #nn.BatchNorm1d(num_features=f),
            nn.ReLU(),
        )
        self.hidden7=nn.Sequential(
            nn.Linear(f,g),
            #nn.BatchNorm1d(num_features=g),
            nn.ReLU(),
        )
        self.hidden8=nn.Sequential(
            nn.Linear(g,h),
            #nn.BatchNorm1d(num_features=h),
            nn.ReLU(),
        )
        self.hidden9=nn.Sequential(
            nn.Linear(h,i),
            #nn.BatchNorm1d(num_features=i),
            nn.ReLU(),
        )
        self.hidden10=nn.Sequential(
            nn.Linear(i,j),
            #nn.BatchNorm1d(num_features=j),
            nn.ReLU(),
        )
        self.regression=nn.Sequential(
            nn.Linear(j,1),
            nn.ReLU(),
        )
    def forward(self,x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.hidden7(x)
        x = self.hidden8(x)
        x = self.hidden9(x)
        x = self.hidden10(x)
        output = self.regression(x)
        return output

def unify(data):
    data = np.array(data)
    mean = np.load('mean.npy')
    scale = np.load('scale.npy')
    unified_data = (data - mean) / scale
    return(unified_data)
x = torch.from_numpy(unify(feature_magpie).astype(np.float32))
mlp = torch.load('feature_extractor.pt')
mlp.eval()
feature_ATL = mlp.hidden10(mlp.hidden9(mlp.hidden8(mlp.hidden7(mlp.hidden6(mlp.hidden5(mlp.hidden4(mlp.hidden3(mlp.hidden2(mlp.hidden1(x)))))))))).detach().numpy()
print(feature_ATL.shape)


# In[134]:


#Expert Knowledge
feature_expert = []
for i in tqdm(range(len(struct))):
    ele = list(struct[i].composition.get_el_amt_dict().keys())
    num = list(struct[i].composition.get_el_amt_dict().values())
    atom_layers,thick = get_atom_layers_thickness(struct[i]) 
    if atom_layers > 1:
        winkle = thick/(atom_layers-1)
    else:
        winkle = 0
    sg = get_space_group_number(struct[i])
    Rnum = dict_Rnum[sg]
    Mnum = dict_Mnum[sg]
    m1,m2,m3 = 0,0,0
    if Mnum == 0:
        m1 = 1
    elif Mnum%2 == 0:
        m2 = 1
    elif Mnum%2 == 1:
        m3 = 1
    HM = dict_Mnum[sg]
    cbfv = make_cbfv([ele,num])
    EN = cbfv[29]-cbfv[30]
    DP = cbfv[61]+cbfv[62]
    ES = cbfv[44]
    EP = cbfv[48]
    ED = cbfv[52]
    feature_vector = [EN,DP,ES,EP,ED,sg,Rnum,Mnum,m1,m2,m3,thick,atom_layers,winkle,HM]
    feature_expert.append(feature_vector)
    #print(ele,num,feature_vector,len(feature_vector))
final_feature = np.concatenate((feature_ATL, np.array(feature_expert)), axis=1)
print(final_feature.shape)


# In[120]:


from joblib import dump,load
model = load('/home/xychen/Carrior Mobility/Code/Final Code/miuE.joblib')
predictions = model.predict(np.array(final_feature, dtype=object))
print(predictions)


# In[ ]:





# In[ ]:




