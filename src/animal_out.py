#!/usr/local/bin/python2.7
# encoding: utf-8
'''
src.animal_out -- Machine Learning Algorithm

@author:     Laercio, Pedro and Lucca
@copyright:  2016 ICMC. All rights reserved.
@license:    license
@contact:    leoabubauru@hotmail.com
@deffield    updated: Updated
'''

import sys
import os
import time
import pandas as datafile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

from argparse                 import ArgumentParser
from argparse                 import RawDescriptionHelpFormatter
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import precision_score
from sklearn.preprocessing    import LabelEncoder, normalize
from sklearn.cross_validation import KFold, cross_val_score


###############################################################################
#                           SOME PARAMETERS
###############################################################################

# Global variables
verbose = 0
nanfill = False
nominal2numeric = False
norm_data = False
run_alg = False
tunning_par = False
choose_alg = False

class rf_param_t:
    n_estimators = 200

class nb_param_t:
    n_estimators = 200

# Target attribute
target_att = ["OutcomeType"]

# ID attribute
id_att = ["ID"]

# Nominal attributes drop out list. Only common attibutes for both train/test files
useless_att_droplist = ["DateTime"]

# features to compute prob
attr_comp = ["singleColor", "singleBreed", "AnimalType", "Sex", "Neutered",\
             "isMix", "hasName", "nbrofColors", "DaysUponOutcome"]


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
       
def get_nan(x):
    if x=='Unknown':
        return np.NaN
    else:
        return x
        
        
###############################################################################
#                    BUILD A NEW TRAIN/TEST FILE FUNCTION
# Each task print some info and calculates spent time by itself.
# Then split some data as the original datafile has mixed info in it.
###############################################################################
def get_new_file(filename):
    global verbose, nanfill
    
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
        if verbose > 0:
            print_progress("Adjusting ID column...")
            start_time = time.clock()
        csv_file=csv_file.rename(columns = {"AnimalID":"ID"})
        csv_file["ID"] = csv_file["ID"].apply(lambda x: x.split("A")[1])
        if verbose > 0:
            print("--> %8.3f seconds" % (time.clock() - start_time))

    # Handling missing values
    if (nanfill == True):
        if verbose > 0:
            print_progress("Filliing NAN with -1...")
            start_time = time.clock()
        csv_file = csv_file.fillna(-1)
    else:    
        if verbose > 0:
            print_progress("Filliing NAN with most frequent value...")
            start_time = time.clock()
        # We convert the csv_file in boolean values. Then we discover which 
        # columns have or not NaN values and iterate over csv_filem to fill 
        # with most frequent value.    
        data_with_nan = csv_file.isnull().any()
        data_with_nan = data_with_nan.drop(["ID", "Name", "DateTime"])
        csv_file_aux = csv_file.dropna()
        for column_with_nan in data_with_nan.index:
            if (data_with_nan[column_with_nan] == True):
                mean_value = csv_file_aux[column_with_nan].value_counts().index[0]
                csv_file[column_with_nan] = csv_file[column_with_nan].fillna(mean_value)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))

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
   
   
###############################################################################
#                           PRE_PROCESS FUNCTION
###############################################################################
def pre_process(csv_file):
    global verbose, nanfill, nominal2numeric, norm_data
 
 
    if verbose > 0:
        print_progress("Removing useless attributes...")
        start_time = time.clock()
    csv_file.drop(useless_att_droplist, axis=1, inplace = True)
    if "OutcomeSubtype" in csv_file.columns:
        csv_file.drop("OutcomeSubtype", axis=1, inplace = True)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))
            

    if (nominal2numeric == True):
        if verbose > 0:
            print_progress("Converting nominal to numeric data...")
            start_time = time.clock()
        to_number = LabelEncoder()
        csv_file["singleColor"] = to_number.fit_transform(csv_file['singleColor'])
        csv_file["singleBreed"] = to_number.fit_transform(csv_file['singleBreed'])
        csv_file["AnimalType"]  = to_number.fit_transform(csv_file['AnimalType'])
        csv_file["Sex"]         = to_number.fit_transform(csv_file['Sex'])
        csv_file["Neutered"]    = to_number.fit_transform(csv_file['Neutered'])
        csv_file["isMix"]       = to_number.fit_transform(csv_file['isMix'])
        csv_file["hasName"]     = to_number.fit_transform(csv_file['hasName'])
        if "OutcomeType" in csv_file.columns:
            csv_file["OutcomeType"] = to_number.fit_transform(csv_file['OutcomeType'])
        if verbose > 0:
            print("--> %8.3f seconds" % (time.clock() - start_time))
            
            # TODO: Vamos implementar a normalização ???
            
        return csv_file
        
        
###############################################################################
#                           RUN_ALGORITHM FUNCTION
###############################################################################
def tunning_parameters():
    parameter = 0


###############################################################################
#                       CHOOSE THE BEST ALGORITHM FUNCTION
###############################################################################
def choose_best_algorithm():
    alg_chosen = ''

    return alg_chosen

###############################################################################
#                           RUN_ALGORITHM FUNCTION
###############################################################################
def run_algorithm(best_alg, train_file, test_file):
    global verbose

    # Gets/Split samples for trainning/test
    if verbose > 0:
        start_time = time.clock()
    if verbose > 0:
        print_progress("Gets/Split samples for trainning/test")
    kf = KFold(len(train_file), n_folds=10, shuffle=False)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))

    if verbose > 0:
        print_progress("Create the random forest object for fitting.")
        start_time = time.clock()
    # random_state=1000 is a magic number. See answer by cacol89 in this question: 
    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html    
    classif = RandomForestClassifier(n_estimators = 200, n_jobs = -1, \
                                     max_features=None, random_state=1000)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    score_result = 0.0
    target = train_file["OutcomeType"].values
    train_data = train_file[attr_comp].values
    for traincv, testcv in kf:
        if verbose > 0:
            print_progress("Performing fitting...")
            start_time = time.clock()
        fit_result = classif.fit(train_data[traincv], target[traincv])
        if verbose > 0:
            print("--> %8.3f seconds" % (time.clock() - start_time))
        
        if verbose > 0:
            print_progress("Calculating training score...")
            start_time = time.clock()
        score = fit_result.score(train_data[testcv], target[testcv])
        if verbose > 0:
            print("--> %8.3f seconds" % (time.clock() - start_time))

        if score_result < score:
            score_result = score
            if verbose > 0:
                print_progress("Performing prediction on test data...")
                start_time = time.clock()
            pred_prob = fit_result.predict_proba(test_file[attr_comp].values)
            if verbose > 0:
                print("--> %8.3f seconds" % (time.clock() - start_time))

    return score_result, test_file["ID"].values, pred_prob


###############################################################################
#                           PRINT PROGRESS FUNCTION
###############################################################################
def print_progress(msg):
# line length
# 1      10        20        30        40        50        60        70       80
# 345678901234567890123456789012345678901234567890123456789012345678901234567890
    msglen = len(msg)
    fillspaces = 50 - msglen 
    msg = msg + fillspaces * ' '
    print ("%s" % msg),    


###############################################################################
#                           SHOW_RESULTS FUNCTION
###############################################################################
def print_results(id_test, pred_prob, training_score, out_filename, totaltime):
    global verbose

    print ("Training accuracy: %.2f" % (training_score * 100.0))    

    if verbose > 0:
        start_time = time.clock()
        print_progress("Writing output file...")
    datafile.DataFrame({"ID"             : id_test, \
                        "Adoption"       : pred_prob[:,0], \
                        "Died"           : pred_prob[:,1], \
                        "Euthanasia"     : pred_prob[:,2], \
                        "Return_to_owner": pred_prob[:,3], \
                        "Transfer"       : pred_prob[:,4]  \
                        }, columns=["ID","Adoption","Died","Euthanasia",\
                                    "Return_to_owner","Transfer"]
                       ).to_csv(out_filename, index=False)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))
        print("Total execution time: %8.3f seconds" % (time.clock() - totaltime))
    if verbose > 0:
        print("Done!")


    
###############################################################################
#                               MAIN FUNCTION
###############################################################################
def main(argv=None): # IGNORE:C0111
    global verbose, nanfill, nominal2numeric, norm_data, run_alg, \
           choose_alg, tunning_par

    total_time = time.clock()

    try:
        # Parser for command line arguments
        parser = ArgumentParser()
        parser.add_argument("-m", dest="norm_data" , default=False, action="store_true", help="normalize numeric data")
        parser.add_argument("-n", dest="nanfill"   , default=False, action="store_true", help="fills NaN values with -1 instead most frequent value")
        parser.add_argument("-v", dest="verbose"   , default=0    , action="count",      help="shows script execution steps")
        parser.add_argument("-x", dest="nom2num"   , default=False, action="store_true", help="convert nominal attributes to numerical")

        # Process arguments
        args           = parser.parse_args()
        verbose        = args.verbose
        train_filename = "../data/train.csv"
        test_filename  = "../data/test.csv"
        out_filename   = "../out/result.csv"
        nanfill        = args.nanfill 
        nominal2numeric= args.nom2num
        norm_data      = args.norm_data

        if verbose > 0:
            print("Verbose mode: ON")

        if (train_filename and test_filename and out_filename) and \
           (train_filename == test_filename) or (train_filename == out_filename) or \
           (test_filename == out_filename):
            print("ERROR: Input and output filenames must be unique!")
            return 1
            
        if (norm_data == True and nominal2numeric == False):
            print("ERROR: To normalize data nominal values must be converted into numbers with -x")
            return 1
        
        # Handle input files as they have mixed info in the attributes
        new_train_file = get_new_file(train_filename)
        new_test_file  = get_new_file(test_filename)
        
        # Pre-process the data
        train_file = pre_process(new_train_file)
        test_file  = pre_process(new_test_file)
        
        # Run cross-validation to tune parameters for the algorithms
        tunning_parameters()
        
        # Run cross-validation to choose the best algorithm for this problem
        best_alg = choose_best_algorithm()
        
        # Run the chosen algorithm
        train_score, id_test, pred_prob = run_algorithm(best_alg, \
                                                        train_file, \
                                                        test_file)
        
        # Print the results and save a file with the probabilities
        print_results(id_test, pred_prob, train_score, out_filename, total_time)        
        
        # Ends application
        return 0
    
    
    # Handle errors
    except Exception as e:
        raise(e)
        return 2


# Application start up
if __name__ == "__main__":
    sys.exit(main())
