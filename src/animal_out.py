#!/usr/local/bin/python2.7
# encoding: utf-8
'''
src.RandomForest -- Machine Learning Algorithm

src.RandomForest is a script to run Random Forest algorithm at a data set

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
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder, normalize

__all__ = []
__version__ = 0.1
__date__ = '2016-06-01'
__updated__ = '2016-06-01'


###############################################################################
#                           SOME PARAMETERS
###############################################################################
DEBUG = 1
verbose = 0
nanfill = False
nominal2numeric = False
norm_data = False
remove_corr = False
run_alg = False
tunning_par = False
choose_alg = False

# Target attribute
target_att = ["target"]

# ID attribute
id_att = ["ID"]

# Nominal attributes drop out list
nominal_att_droplist = ["v3", "v22", "v24", "v30", "v31", "v47", "v52", "v56", \
                        "v66", "v71", "v74", "v75", "v79", "v91", "v107", \
                        "v110", "v112", "v113", "v125"]

# No-distribution attributes drop out list
nodist_att_droplist = ["v23", "v38"]

# Const-distribution attributes drop out list
constdist_att_droplist = []

# Duplicated attributes drop out list. Leaving only one of them.
double_att_droplist = []

# Attributes with correlation > 95% drop out list
correlation95_att_droplist = ["v46", "v53", "v54", "v60", "v63", "v76", "v83", \
                              "v89", "v95", "v96", "v100","v105", "v106","v114",\
                              "v115","v116","v118","v121",]

# Others attributes drop out list
others_att_droplist = []


###############################################################################
#                           EXCEPTION CLASS
###############################################################################
class CLIError(Exception):
    '''Generic exception to raise and log different fatal errors.'''
    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "E: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg
        
        
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
#                       BUILD A NEW TRAIN FILE FUNCTION
###############################################################################
def get_new_file(filename):
    global verbose

    if verbose > 0:
        print_progress("Opening %s file to rebuild it." % os.path.abspath(filename))
        start_time = time.clock()
    csv_file = datafile.read_csv(filename)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))


    if verbose > 0:
        print_progress("Converting age to days...")
        start_time = time.clock()
    feature_values = csv_file["AgeuponOutcome"].values
    csv_file["DaysUponOutcome"] = age_to_days(feature_values)
    csv_file.drop("AgeuponOutcome", axis=1, inplace = True)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))

    
    if verbose > 0:
        print_progress("Splitting sex and neutered info...")
        start_time = time.clock()
    csv_file["Sex"] = csv_file["SexuponOutcome"].apply(get_sex)
    csv_file["Neutered"] = csv_file["SexuponOutcome"].apply(get_neutered)
    csv_file.drop("SexuponOutcome", axis=1, inplace = True)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))


    if verbose > 0:
        print_progress("Splitting date and time info...")
        start_time = time.clock()
    csv_file["Date"] = csv_file["DateTime"].apply(get_date_info)
    csv_file["Time"] = csv_file["DateTime"].apply(get_time_info)
    csv_file.drop("DateTime", axis=1, inplace = True)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))


    if verbose > 0:
        print_progress("Detecting if is a Mix breed...")
        start_time = time.clock()
    csv_file["isMix"] = csv_file["Breed"].apply(lambda x: "Mix" in x)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))


    if verbose > 0:
        print_progress("Getting first breed and removing Mix...")
        start_time = time.clock()
    csv_file["singleBreed"] = csv_file["Breed"].apply(lambda x: x.split("/")[0])
    csv_file["singleBreed"] = csv_file["singleBreed"].apply(lambda x: x.split(" Mix")[0])
    csv_file.drop("Breed", axis=1, inplace = True)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))

    
    if verbose > 0:
        print_progress("Getting first color...")
        start_time = time.clock()
    csv_file["singleColor"] = csv_file["Color"].apply(lambda x: x.split("/")[0])
    csv_file.drop("Color", axis=1, inplace = True)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))
    
    return csv_file    
   
   
###############################################################################
#                           PRE_PROCESS FUNCTION
###############################################################################
def pre_process(filename):
    global verbose, nanfill, nominal2numeric, norm_data, remove_corr
 
    if verbose > 0:
        print_progress("Opening file to pre-process: %s" % os.path.abspath(filename))
        start_time = time.clock()
    csv_file = datafile.read_csv(filename)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))
 
 
#     if (nominal2numeric == True):
#         if verbose > 0:
#             print_progress("Converting nominal to numeric data...")
#             start_time = time.clock()
#         to_number = LabelEncoder()
#         csv_file["v3"  ] = to_number.fit_transform(csv_file.v3)
#         csv_file["v22" ] = to_number.fit_transform(csv_file.v22)
#         csv_file["v24" ] = to_number.fit_transform(csv_file.v24)
#         csv_file["v30" ] = to_number.fit_transform(csv_file.v30)
#         csv_file["v31" ] = to_number.fit_transform(csv_file.v31)
#         csv_file["v47" ] = to_number.fit_transform(csv_file.v47)
#         csv_file["v52" ] = to_number.fit_transform(csv_file.v52)
#         csv_file["v56" ] = to_number.fit_transform(csv_file.v56)
#         csv_file["v66" ] = to_number.fit_transform(csv_file.v66)
#         csv_file["v71" ] = to_number.fit_transform(csv_file.v71)
#         csv_file["v74" ] = to_number.fit_transform(csv_file.v74)
#         csv_file["v75" ] = to_number.fit_transform(csv_file.v75)
#         csv_file["v79" ] = to_number.fit_transform(csv_file.v79)
#         csv_file["v91" ] = to_number.fit_transform(csv_file.v91)
#         csv_file["v107"] = to_number.fit_transform(csv_file.v107)
#         csv_file["v110"] = to_number.fit_transform(csv_file.v110)
#         csv_file["v112"] = to_number.fit_transform(csv_file.v112)
#         csv_file["v113"] = to_number.fit_transform(csv_file.v113)
#         csv_file["v125"] = to_number.fit_transform(csv_file.v125)
#         if verbose > 0:
#             print("--> %8.3f seconds" % (time.clock() - start_time))
# 
# 
#     if (nominal2numeric == False):        
#         if verbose > 0:
#             print_progress("Removing nominal attributes...")
#             start_time = time.clock()
#         csv_file.drop(nominal_att_droplist, axis=1, inplace = True)
#         if verbose > 0:
#             print("--> %8.3f seconds" % (time.clock() - start_time))
#         
#     if (remove_corr == True):
#         if verbose > 0:
#             print_progress("Removing attributes with correlation >= 95% ...")
#             start_time = time.clock()
#         csv_file.drop(correlation95_att_droplist, axis=1, inplace = True)
#         if verbose > 0:
#             print("--> %8.3f seconds" % (time.clock() - start_time))
#    
#     # Only remove lines for training. Test data must be treated with all data. 
#     if verbose > 0:
#         start_time = time.clock()
#     if (nanfill == True):
#         if verbose > 0:
#             print_progress("Filliing NAN with -1...")
#         processed_file = csv_file.fillna(-1)
#     else:
#         if verbose > 0:
#             print_progress("Removing NAN from data...")
#         processed_file = csv_file.dropna()
#     if verbose > 0:
#         print("--> %8.3f seconds" % (time.clock() - start_time))
# 
#     # processed_file still keep 'ID' and, maybe, 'target' attributes.
#     # Let's remove them!
#     id_data = processed_file["ID"].values
#     if "target" in csv_file.columns:
#         target_data    = processed_file["target"].values
#         processed_file = processed_file.drop(target_att + id_att, axis=1)
#         if (norm_data == True):
#             if verbose > 0:
#                 print_progress("Normalizing data...")
#                 start_time = time.clock()
#             processed_file = normalize(processed_file, norm='l2', axis=1, copy=False)
#             if verbose > 0:
#                 print("--> %8.3f seconds" % (time.clock() - start_time))
#         return processed_file, id_data, target_data
#     else:
#         processed_file = processed_file.drop(id_att, axis=1)
#         if (norm_data == True):
#             if verbose > 0:
#                 print_progress("Normalizing data...")
#                 start_time = time.clock()
#             processed_file = normalize(processed_file, norm='l2', axis=1, copy=False)
#             if verbose > 0:
#                 print("--> %8.3f seconds" % (time.clock() - start_time))
        # TODO Remover as atribuicoes abaixo
        processed_file = csv_file.dropna()
        id_data = processed_file["ID"].values
        return processed_file, id_data
        
        
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
def run_algorithm(best_alg, train_file, test_file, target_data, train_sample_size):
#     global verbose
#     perc = 0.1 # Percentage to build a training/test files
#     
#     if verbose > 0:
#         print_progress("Create the random forest object for fitting.")
#         start_time = time.clock()
#     # random_state=1000 is a magic number. See answer by cacol89 in this question: 
#     # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html    
#     classif = RandomForestClassifier(n_estimators = 200, n_jobs = -1, \
#                                      max_features=None, random_state=1000, \
#                                      class_weight={1:0.7612, 0:0.2388})
#     if verbose > 0:
#         print("--> %8.3f seconds" % (time.clock() - start_time))
# 
# 
#     if verbose > 0:
#         print_progress("Creating training data for fitting...")
#         start_time = time.clock()
#     # We need a subset of a known data in order to fit the classifier
#     # and calculate its score.
#     train_fit_file  = train_file[:train_sample_size]
#     fit_target_data = target_data[:train_sample_size]
#     if verbose > 0:
#         print("--> %8.3f seconds" % (time.clock() - start_time))
# 
#     if verbose > 0:
#         print_progress("Performing fitting...")
#         start_time = time.clock()
#     fit_result = classif.fit(train_fit_file, fit_target_data)
#     if verbose > 0:
#         print("--> %8.3f seconds" % (time.clock() - start_time))
#     
#     
#     if verbose > 0:
#         print_progress("Performing prediction on training data...")
#         start_time = time.clock()
#     # As we took a percentage of data to fit the classifier, now we use 
#     # 100% - perc for training data
#     train_file  = train_file [int(len(fit_target_data)):int(len(fit_target_data))+int(len(fit_target_data) * perc)]
#     target_data = target_data[int(len(fit_target_data)):int(len(fit_target_data))+int(len(fit_target_data) * perc)]
#     prediction  = fit_result.predict(train_file)    
#     if verbose > 0:
#         print("--> %8.3f seconds" % (time.clock() - start_time))
# 
# 
#     if verbose > 0:
#         print_progress("Calculating training score...")
#         start_time = time.clock()
#     training_score = precision_score(target_data, prediction)
#     if verbose > 0:
#         print("--> %8.3f seconds" % (time.clock() - start_time))
#     
# 
#     if verbose > 0:
#         print_progress("Performing prediction on test data...")
#         start_time = time.clock()
#     prediction  = fit_result.predict(test_file)    
#     pred_prob   = fit_result.predict_proba(test_file)
#     if verbose > 0:
#         print("--> %8.3f seconds" % (time.clock() - start_time))
# 
# 
    # TODO Remover as atribuicoes abaixo
    training_score = 0.0
    pred_prob = 0.0
    prediction = 0.0
    return training_score, pred_prob, prediction


###############################################################################
#                           SHOW_RESULTS FUNCTION
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
def show_results(pred_prob, prediction, training_score, id_test, out_filename, totaltime):
    global verbose

    print ("Precision training score: %.2f" % (training_score * 100.0))    

    if verbose > 0:
        start_time = time.clock()
        print_progress("Wrinting output file...")
    datafile.DataFrame({"ID": id_test, "PredictedProb": pred_prob[:,1]}).\
                        to_csv(out_filename, index=False)
    if verbose > 0:
        print("--> %8.3f seconds" % (time.clock() - start_time))
        print("Total execution time: %8.3f seconds" % (time.clock() - totaltime))
    if verbose > 0:
        print("Done!")


    
###############################################################################
#                               MAIN FUNCTION
###############################################################################
def main(argv=None): # IGNORE:C0111
    global verbose, nanfill, nominal2numeric, norm_data, remove_corr, run_alg, \
           choose_alg, tunning_par

    total_time = time.clock()

    '''Command line options.'''
    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = '''%s

  Created by Laercio Barbosa on %s.
  Copyright 2016 ICMC. All rights reserved.

  Licensed under the Apache License 2.0
  http://www.apache.org/licenses/LICENSE-2.0

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument("-c", dest="remove_cor", default=False, action="store_true", help="remove attributes with correlation >= 95% between each other")
        parser.add_argument("-m", dest="norm_data" , default=False, action="store_true", help="norm numeric data")
        parser.add_argument("-n", dest="nanfill"   , default=False, action="store_true", help="fills nan values with -1")
        parser.add_argument("-s", dest="size_tr"   , default=1000 ,                      help="sample size for training")
        parser.add_argument("-v", dest="verbose"   , default=0    , action="count",      help="shows script execution steps")
        parser.add_argument("-x", dest="nom2num"   , default=False, action="store_true", help="convert nominal attributes to numerical")

        # Process arguments
        args           = parser.parse_args()
        verbose        = args.verbose
        train_filename = "./data/train.csv"
        test_filename  = "./data/test.csv"
        out_filename   = "./out/result.csv"
        nanfill        = args.nanfill 
        nominal2numeric= args.nom2num
        norm_data      = args.norm_data
        remove_corr    = args.remove_cor
        sample_size_tr = int(args.size_tr)

        if verbose > 0:
            print("Verbose mode: ON")

        if (train_filename and test_filename and out_filename) and \
           (train_filename == test_filename) or (train_filename == out_filename) or \
           (test_filename == out_filename):
            raise CLIError("Input and output filenames must be unique!")
        
        
        # Let's play and have some fun!
        new_train_file = get_new_file(train_filename)
        new_test_file  = get_new_file(test_filename)
        
        train_file, _, target_data = pre_process(new_train_file)
        test_file, test_id_data    = pre_process(new_test_file)
        
        tunning_parameters()
        best_alg = choose_best_algorithm()
        train_score, pred_prob, prediction = run_algorithm(best_alg, \
                                                           train_file, \
                                                           test_file, \
                                                           target_data, \
                                                           sample_size_tr)
        show_results(pred_prob, prediction, train_score, test_id_data, \
                     out_filename, total_time)        
        
        
        return 0
    except Exception, e:
        if DEBUG:
            raise(e)
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help, use --help")
        return 2


if __name__ == "__main__":
    sys.exit(main())
