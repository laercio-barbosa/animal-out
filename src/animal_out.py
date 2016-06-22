# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 01:27:47 2016

@author: PeDeNRiQue
"""
import sys
import os
import time
import pandas as datafile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.svm import SVC

###############################################################################
#                           SOME PARAMETERS
###############################################################################

# Global variables
verbose = 0
nanfill = False
nominal2numeric = False
norm_data = False
remove_corr = False
run_alg = False
tunning_par = False
choose_alg = False
new_train_file = ""
clf = ""
result = ""

###############################################################################
#                       AGE CONVERSION FUNCTION
###############################################################################
def age_to_days(item):
    # convert item to list if it is one string
    if type(item) is str:
        item = [item]
    ages_in_days = datafile.np.zeros(len(item))
    for i in range(len(item)):
        # check if item[i] is str
        if type(item[i]) is str:
            if 'day' in item[i]:
                ages_in_days[i] = int(item[i].split(' ')[0])
            if 'week' in item[i]:
                ages_in_days[i] = int(item[i].split(' ')[0])*7
            if 'month' in item[i]:
                ages_in_days[i] = int(item[i].split(' ')[0])*30
            if 'year' in item[i]:
                ages_in_days[i] = int(item[i].split(' ')[0])*365    
        else:
            # item[i] is not a string but a nan
            ages_in_days[i] = 0
    return ages_in_days


###############################################################################
#                       GET SEX FUNCTION
###############################################################################
def get_sex(x):
    x = str(x)
    if x.find('Male')   >= 0: return 'male'
    if x.find('Female') >= 0: return 'female'
    return 'unknown'


###############################################################################
#                       GET NEUTERED FUNCTION
###############################################################################
def get_neutered(x):
    x = str(x)
    if x.find('Spayed')   >= 0: return 'neutered'
    if x.find('Neutered') >= 0: return 'neutered'
    if x.find('Intact')   >= 0: return 'intact'
    return 'unknown'
        
        
###############################################################################
#                       GET DATE FUNCTION
###############################################################################
def get_date_info(date_time):
    date_time = str(date_time)
    return date_time.split(" ")[0]
        
       
###############################################################################
#                       GET TIME FUNCTION
###############################################################################
def get_time_info(date_time):
    date_time = str(date_time)
    return date_time.split(" ")[1]
    
 
 ###############################################################################
#                    BUILD A NEW TRAIN/TEST FILE FUNCTION
# Each task print some info and calculates spent time by itself.
# Then split some data as the original datafile has mixed info in it.
###############################################################################
def get_new_file(filename):
    global verbose
    
    # First of all we need open/read the datafile
    if verbose > 0:
        print_progress("Opening %s file to rebuild it." % os.path.abspath(filename))
        start_time = time.clock()
    csv_file = datafile.read_csv(filename)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))
        
    # One of the files has a different column ID name. Fix it so that
    # both train and test have the same column name for ID    
    if "AnimalID" in csv_file.columns:
        csv_file=csv_file.rename(columns = {"AnimalID":"ID"})


    # Then we convert 'AgeuponOutcome' to unit 'days'
    if verbose > 0:
        print_progress("Converting age to days...")
        start_time = time.clock()
    feature_values = csv_file["AgeuponOutcome"].values
    csv_file["DaysUponOutcome"] = age_to_days(feature_values)
    csv_file.drop("AgeuponOutcome", axis=1, inplace = True)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))

    
    # Split sex and neutered info in two new columns
    if verbose > 0:
        print_progress("Splitting sex and neutered info...")
        start_time = time.clock()
    csv_file["Sex"] = csv_file["SexuponOutcome"].apply(get_sex)
    csv_file["Neutered"] = csv_file["SexuponOutcome"].apply(get_neutered)
    csv_file.drop("SexuponOutcome", axis=1, inplace = True)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))


    # Date/time is also splited in two new columns
    if verbose > 0:
        print_progress("Splitting date and time info...")
        start_time = time.clock()
    csv_file["Date"] = csv_file["DateTime"].apply(get_date_info)
    csv_file["Time"] = csv_file["DateTime"].apply(get_time_info)
    csv_file.drop("DateTime", axis=1, inplace = True)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))


    # Generates a new column with boolean info 'isMix' for breed
    if verbose > 0:
        print_progress("Detecting if is a Mix breed...")
        start_time = time.clock()
    csv_file["isMix"] = csv_file["Breed"].apply(lambda x: "Mix" in x)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))


    # Breed must be handled as it has many different types. So we
    # take only the first breed before '/' and remove 'Mix'
    if verbose > 0:
        print_progress("Getting first breed and removing Mix...")
        start_time = time.clock()
    csv_file["singleBreed"] = csv_file["Breed"].apply(lambda x: x.split("/")[0])
    csv_file["singleBreed"] = csv_file["singleBreed"].apply(lambda x: x.split(" Mix")[0])
    csv_file.drop("Breed", axis=1, inplace = True)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))

    
    # Also for colors we split them and take only the first one
    if verbose > 0:
        print_progress("Getting first color...")
        start_time = time.clock()
    csv_file["singleColor"] = csv_file["Color"].apply(lambda x: x.split("/")[0])
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))
        
        
    # Count colors in each animal
    if verbose > 0:
        print_progress("Counting color for each animal ...")
        start_time = time.clock()
    csv_file["nbrofColors"] = csv_file["Color"].apply(lambda x: len((x.split("/"))))
    csv_file.drop("Color", axis=1, inplace = True)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))
        
                            
    # Create a atribute with info if the animal has a name
    if verbose > 0:
        print_progress("Has the animal a name?")
        start_time = time.clock()
    csv_file["hasName"] = csv_file["Name"].apply(lambda x: type(x) is str)
    csv_file.drop("Name", axis=1, inplace = True)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))
        
        
    # Now we have a new datafile
    return csv_file    

def pre_process_shelter(csv_file):
    to_number = LabelEncoder()
    csv_file["AnimalType"] = to_number.fit_transform(csv_file['AnimalType'])
    csv_file["OutcomeType"] = to_number.fit_transform(csv_file['OutcomeType'])
    csv_file["Sex"] = to_number.fit_transform(csv_file['Sex'])
    csv_file["Neutered"] = to_number.fit_transform(csv_file['Neutered'])
    csv_file["isMix"] = to_number.fit_transform(csv_file['isMix'])
    csv_file["hasName"] = to_number.fit_transform(csv_file['hasName'])

    return csv_file

def main(argv=None):
    global new_train_file
    
    actual_directory = os.path.dirname(os.path.abspath(__file__))
    
    train_file = actual_directory+'\\data\\train.csv';
            
    records = datafile.read_csv(train_file)
    
    #print (records['OutcomeType'])
    
    new_train_file = get_new_file(train_file)
    new_train_file = pre_process_shelter(new_train_file)
    
 
    
    new_train_file.drop("ID", axis=1, inplace = True)
    new_train_file.drop("OutcomeSubtype", axis=1, inplace = True)
    new_train_file.drop("Date", axis=1, inplace = True)
    new_train_file.drop("Time", axis=1, inplace = True)
    new_train_file.drop("singleBreed", axis=1, inplace = True)
    new_train_file.drop("singleColor", axis=1, inplace = True)
    
    target = new_train_file["OutcomeType"]
    new_train_file.drop("OutcomeType", axis=1, inplace = True)
    
    train_file = new_train_file[:21000]
    train_target = target[:21000]
    
    test_file = new_train_file[-5729:]    
    
    clf = SVC(probability=True)
    clf.fit(np.array(train_file), np.array(train_target)) 
    
    result = clf.predict_proba(test_file)
    #result = clf.predict(test_file)
    
    print(target[-6729:])
    print(result)

    np.savetxt("Result.txt", result, fmt='%d', delimiter=' ', newline='\n', header='', footer='', comments='# ')
    np.savetxt("Target.txt", target, fmt='%d', delimiter=' ', newline='\n', header='', footer='', comments='# ')

    print ("FIM")
    return 0
    
    
# Application start up
if __name__ == "__main__":
    sys.exit(main())