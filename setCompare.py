# -*- coding: utf-8 -*-
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import numpy as np
import random
from matplotlib import pylab as plt

"""
Michael Mulcahy
Created: 02/04/16
Updated: 02/22/16
Compare set of individual molecules against an SD file using Nick Wood's pythagorean set comparison method. Designed specifically for the Ames mutagenic set.
"""


def setCompare(setDir='',xi=0.0001,model='Tanimoto',CV=5): # setCompare(sdf-File Locaiton, xi, model type, and cross validation set to be used as test set)
    
    sdfSet,CasNoSet,actSet = genSet(setDir) # Call set generator
    
    # Initialize test set and associated test set data
    # In the future, explore creating a class to handle all of this data?
    tSet = list(sdfSet[CV+1])       # Test set
    del sdfSet[CV+1]                # Remove test set from sd matrix
    actPH = list(actSet[CV+1])      # Activity place holder of test set
    del actSet[CV+1]                # Remove test set from activity matrix
    CasNoSet = list(CasNoSet[CV+1]) # CAS numbers for test set
    
    # Re-organize sd array into inactive and active molecule arrays
    for i in range(2,len(sdfSet)):
        for j in range(len(sdfSet[i])):
            sdfSet[actSet[i][j]].append(sdfSet[i][j])
    
    del sdfSet[2:]          # Remove now redundant arrays
    actSet = list(actPH)    # Initialize test set activity
    del actPH
    
    # Remove duplicate and zero valued explicit bit vectors
    sdfSet = [sdfRemoveDup(sdfSet[i]) for i in range(len(sdfSet))]
    
    # Match active and inactive training set lengths
    minLenSdf = min([len(sdfSet[i]) for i in range(len(sdfSet))])               # Get minimum array length
    sdfSet = [random.sample(sdfSet[i],minLenSdf) for i in range(len(sdfSet))]   # Choose random molecules to remove from arrays
        
    # Generate set lengths
    lenSdf = [len(sdfSet[i]) for i in range(len(sdfSet))]
    lenT = len(tSet)

    # Generate similarity comparison and training arrays
    simT = [[[DataStructs.FingerprintSimilarity(tSet[i],sdfSet[j][k]) for k in range(lenSdf[j])] for i in range(lenT)] for j in range(len(sdfSet))]                             # Compare each test molecule against the training set
    simSdf = [np.linalg.inv([[DataStructs.FingerprintSimilarity(sdfSet[i][j],sdfSet[i][k]) for j in range(lenSdf[i])] for k in range(lenSdf[i])]) for i in range(len(sdfSet))]  # Compare training set molecules against themselves
    
    # Calculate simularity
    simTtoSDF = [[np.dot(simT[i][j],np.dot(simSdf[i],np.transpose(simT[i][j]))) for j in range(lenT)] for i in range(len(sdfSet))]
        
    # Generate molecular assignment dictionary
    molDict={}      # Dictionary
    molAssign=[]    # Assignment bit vector (0 = inactive, 1 = active)
    for i in range(lenT):
        if simTtoSDF[1][i] - simTtoSDF[0][i] > xi:
            molAssign.append(1)
            molDict.update({CasNoSet[i]:1})
        elif simTtoSDF[1][i] - simTtoSDF[0][i] < -xi:
            molAssign.append(0)
            molDict.update({CasNoSet[i]:0})
        else:
            molAssign.append('NA')
            molDict.update({CasNoSet[i]:'NA'})
            
    # Generate arrays for easy graphing
    assignGraph=[[],[]]
    for i in range(lenT):
        ds = simTtoSDF[1][i] - simTtoSDF[0][i]  # Difference in active and inactive simularity values
        if  actSet[i]==1:
            assignGraph[1].append(ds)           # Positive assignment array
        elif actSet[i]==0:
            assignGraph[0].append(ds)           # Negative assignment array
    
    # Generate Figures
    f = plt.figure(1) # figure 1
    
    # Subplot 1 - Predicted Ames mutagen positive plot
    f1 = plt.subplot(121) # Subplot 1 (1 row 2 columns)
    plt.plot(assignGraph[1],'g.',[0,len(assignGraph[1])],[-xi,-xi],'b-',[0,len(assignGraph[1])],[xi,xi],'b-')
    plt.title('Ames Mutagen + CV5')
    f1.axes.get_xaxis().set_ticks([])
    plt.ylabel('${\sigma}$(+) - ${\sigma}$(-)')
    
    # Subplot 2 - Predicted Ames mutagen negative plot
    f2 = plt.subplot(122, sharey=f1) # Subplot 2 (1 row 2 columns)
    plt.plot(assignGraph[0],'r.',[0,len(assignGraph[0])],[-xi,-xi],'b-',[0,len(assignGraph[0])],[xi,xi],'b-')
    plt.title('Ames Mutagen - CV5')
    f2.axes.get_xaxis().set_ticks([])
    
    # Save figure 1
    f.savefig('C:/Users/Michael/Documents/Research/CV5.pdf')
    acc,spe,sen,con = genReport(molAssign,actSet)
    return acc,spe,sen,con,molDict
        
"""
Michael Mulcahy
Created: 02/10/16
Modified 02/10/16
sdfRemoveDup removes duplicate or all 0 valued MACCS keys explicit bit vectors from the sd array.
"""    
def sdfRemoveDup(sdfSetMK):
    i=0
    while i<len(sdfSetMK)-1:
        dt = sdfSetMK[i]
        if dt in sdfSetMK[(i+1):] or dt.GetNumOnBits==0:
            del sdfSetMK[i]
        else:
            i=i+1
    
    """
    while i<len(sdfSetMK)-1:
        dt = sdfSetMK[i]
        simT = [DataStructs.FingerprintSimilarity(dt,sdfSetMK[j]) for j in range(1,len(sdfSetMK))]
        if any(simT) > 0.999:
            del dt
        else:
            i=i+1
    """
    return (sdfSetMK)
    
    
"""
Michael Mulcahy
Created: 02/15/16
Updated: 02/15/16
genSet identifies and separates each specific set into their respective cross-validation or training sets
"""    
def genSet(setDir=''):
    supplSet = Chem.SupplierFromFilename(setDir)
    fil_in = open(setDir,'r')
    fil_line = fil_in.readlines()
    
    # Information Sets
    # FUTURE - Change range to len(unique identifiers) where unique identifier is set name (training, cv1, cv2, etc...)
    sdfSet = [[] for i in range(7)]     # MACCS keys for each unique set
    CasNoSet = [[] for i in range(7)]   # Cas numbers for each unique set
    actSet=[[] for i in range(7)]       # Activity for each unique set

    # Value Place Holders and counter
    CasNo=''
    Activity=''
    Set=''
    j=0

    # Parse SD file for information
    for i in range(len(fil_line)):
        if '$$$$' in fil_line[i]:   # $$$$ is molecule break identifier
            if Set == 'TRAIN':
                sdfSet[Activity].append(MACCSkeys.GenMACCSKeys(supplSet[j]))
                CasNoSet[Activity].append(CasNo)
                actSet[Activity].append(Activity)
            else:
                sdfSet[Set].append(MACCSkeys.GenMACCSKeys(supplSet[j]))
                CasNoSet[Set].append(CasNo)
                actSet[Set].append(Activity)
            j=j+1
        elif '> <CAS_NO>' in fil_line[i]:
            CasNo = fil_line[i+1][:-1]
        elif '> <Activity>' in fil_line[i]:
            Activity = int(fil_line[i+1][0])
        elif '> <Set>' in fil_line[i]:
            if 'CV' in fil_line[i+1]:
                Set = (int(fil_line[i+1][2]))+1
            else:
                Set = 'TRAIN'
    return sdfSet,CasNoSet,actSet
    
    
"""
Michael Mulcahy
Created: 02/20/16
Updated: 02/20/16
genReport analyzes the assignment data and produces accuracy, specificity, sensitivity, and confused molecules
"""  

def genReport(molAssign,cActSet):
    report = np.array([float(0),float(0),float(0),float(0),float(0)]) # [NA, TN, TP, FN, FP]
    for i in range(len(cActSet)):
        if molAssign[i] == 'NA':
            report[0]=report[0]+1
        elif molAssign[i]==cActSet[i] and molAssign[i]==0:
            report[1]=report[1]+1
        elif molAssign[i]==cActSet[i] and molAssign[i]==1:    
            report[2]=report[2]+1
        elif molAssign[i]!=cActSet[i] and molAssign[i]==0:
            report[3]=report[3]+1
        elif molAssign[i]!=cActSet[i] and molAssign[i]==1:
            report[4]=report[4]+1
    acc = (report[1]+report[2])/(np.sum(report[1:]))
    spe = report[1]/(report[1]+report[4])
    sen = report[2]/(report[2]+report[3])
    con = report[0]
    return acc,spe,sen,con