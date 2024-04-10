import json
import pandas as pd 
import random 
import math 
import copy 
import os 
import numpy as np 

from sklearn import metrics 

#Help from here: https://analyticsindiamag.com/evaluation-metrics-in-ml-ai-for-classification-problems-wpython-code/


#This file is for making a predictor based on a rule. 
#df = pd.read_csv("frost_csvs/npp_c_cali.csv")

def load_rules(filename):
    with open(filename) as f:
        rules_list = json.load(f)
    return rules_list
    #print(json.dumps(rules_list, indent=4))



def get_outlier_sequence_bounds(which, min_max, rule):
        curr_bound = None 
        parameters_dict = rule["parameters"]
        for item in list(rule["parameters"].keys()):
            sub_latest = parameters_dict[item]["seq_lower_bound"]
            sub_earliest = parameters_dict[item]["seq_upper_bound"]
            if which == "upper":
                bound_of_interest = sub_earliest
            else:
                bound_of_interest = sub_latest
            if curr_bound == None:
                curr_bound = bound_of_interest
            else:
                if min_max == "min":
                    if bound_of_interest < curr_bound:
                        curr_bound = bound_of_interest
                elif min_max == "max":
                    if bound_of_interest > curr_bound:
                        curr_bound = bound_of_interest
        return curr_bound


def build_param_specific_query(param_name, rule):
    parameters_dict = rule["parameters"]
    lower = parameters_dict[param_name]["lower_bound"]
    upper = parameters_dict[param_name]["upper_bound"]
    query_string = f'{param_name} >= {lower} & {param_name} <= {upper}'
    return query_string


def get_indexes(param_name, rule, df):
    query = build_param_specific_query(param_name, rule)
    #print(query)
    bool_df = df.eval(query)
    indexes = bool_df[bool_df].index
    index_list = indexes.tolist()
    return index_list

def build_fulfilment_indexes(param_name, param_indexes, rule, len_df):
        overall_list = []
        parameters_dict = rule["parameters"]
        lower = parameters_dict[param_name]["seq_lower_bound"]
        upper = parameters_dict[param_name]["seq_upper_bound"]
        adding_list = list(range(lower, upper+1))
        #print("Adding List")
        #print(adding_list)
        fulfilled_indexes = None
        #print("Raw Indexes")
        #print(param_indexes)
        first = True
        for add_val in adding_list:
            raw_indexes = np.array(param_indexes)
            added_indexes = raw_indexes + add_val
            if first:
                fulfilled_indexes = added_indexes
                first=False
            else:
                fulfilled_indexes = np.concatenate((fulfilled_indexes, added_indexes), axis=0)
        final = np.unique(fulfilled_indexes)

        final = final[final < len_df]
        final = final[final >= 0]
        #print("Final Fulfulled")
        #print(final)
        return final

def build_rule_prediction_query(rule):
    parameters_dict = rule["parameters"]
    query_string = ''
    first = 1
    for param in list(parameters_dict.keys()):
        lower = parameters_dict[param]["lower_bound"]
        upper = parameters_dict[param]["upper_bound"]
        if not first:
            query_string = query_string + ' & '
        query_string = query_string + f'{param} >= {lower} & {param} <= {upper}'
        #print(query_string)
        first = 0
    return query_string



#Takes in a rule, a dataframe (to predict on), and returns predictions.
def get_predictions_from_rule(rule, test_df, sequence=False):
    if sequence:
        final_indexes = None
        first = True
        for param_name in list(rule["parameters"].keys()):
            #Get the indexes where the parameters are between those values
            param_indexes = get_indexes(param_name, rule, test_df)
            #Get the indexes that the parameters alone would fulfill as potential consequents
            fulfilled_indexes = build_fulfilment_indexes(param_name, param_indexes, rule, len(test_df.index))
            #Return the intersection of these and the existing fulfilled parameters. Must be in all. 
            if first:
                final_indexes = fulfilled_indexes
                first = False
            else:
                final_indexes = np.intersect1d(final_indexes, fulfilled_indexes, assume_unique=True)
            #If it's ever the case the a new list doesn't have something in common with the current one:
            if final_indexes.size == 0:
                break 
        predict_df = test_df.assign(predictions=0)
        for index in final_indexes:
            predict_df.at[index, "predictions"] = 1
        predict_df.fillna(0, inplace=True)
        first_valid_index = get_outlier_sequence_bounds("lower", "max", rule)
    else:
        query = build_rule_prediction_query(rule)
        predict_df = test_df.assign(predictions=test_df.eval(query))
        predict_df.fillna(0, inplace=True)
        predict_df["predictions"] = predict_df["predictions"].astype(int)
        first_valid_index = False
    return predict_df, first_valid_index


def get_empty_eval_dict():
    eval_dict = {
        "Rule Index": [],
        "Accuracy": [],
        "True_Negatives": [],
        "False_Positives": [],
        "False_Negatives": [],
        "True_Positives": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": []
    }
    return eval_dict.copy()



#Evaluate the prediction model 
def evaluate_prediction_model(predict_df, key, model_index=0, first_valid_index=False):
    #Change here - Sanity Checks 
    #predict_df.to_csv("Testing.csv")
    eval_dict = {}
    eval_dict["Rule Index"] = model_index
    #print(first_valid_index)
    if first_valid_index:
        eval_df = predict_df.iloc[first_valid_index:]
    else:
        eval_df = predict_df

    pred = eval_df["predictions"].values.tolist()
    true = eval_df[key].values.tolist()
   
    eval_dict["Accuracy"] = metrics.accuracy_score(true, pred)
    confusion_matrix = metrics.confusion_matrix(true, pred)
    values_array = confusion_matrix.ravel()
    if len(values_array) >1:
        eval_dict["True_Negatives"] = values_array[0]
        eval_dict["False_Positives"] = values_array[1] 
        eval_dict["False_Negatives"] = values_array[2]
        eval_dict["True_Positives"] = values_array[3]
    else:
        print(values_array)
    eval_dict["Precision"] = metrics.precision_score(true, pred, pos_label=1)
    eval_dict["Recall"] = metrics.recall_score(true, pred, pos_label=1)
    eval_dict["F1 Score"] = metrics.f1_score(true, pred, pos_label=1)
    return eval_dict

#This is a bit screwed up for average predictions 
def ensemble_learn(list_of_rules, test_df, sequence=False):
    #Get the predictions for each rule in the list
    num_models = len(list_of_rules)
    prediction_list = []
    valid_indexes = []
    #Get all the prediction dfs for a single rule 
    for single_rule in list_of_rules:
        sub_df, first_valid_index = get_predictions_from_rule(single_rule, test_df, sequence=sequence)
        valid_indexes.append(first_valid_index)
        #Weight them appropriately
        prediction_list.append(sub_df)

    num_predictors = len(list_of_rules)
    #Simple majority vote - more than half wins 
    vote_threshold = num_predictors/2 
    first_predictions =  prediction_list[0]
    for i in range(1, len(prediction_list)):
        first_predictions["predictions"] = first_predictions["predictions"] + prediction_list[i]["predictions"]
    
    first_predictions.loc[first_predictions["predictions"] >= vote_threshold, "predictions"] = 1
    first_predictions.loc[first_predictions["predictions"] < vote_threshold, "predictions"] = 0
    return first_predictions, min(valid_indexes)

def ensemble_learn_or(list_of_rules, test_df, sequence=False):
    #Get the predictions for each rule in the list
    num_models = len(list_of_rules)
    prediction_list = []
    #Get all the prediction dfs for a single rule 
    valid_indexes = []
    for single_rule in list_of_rules:
        sub_df, first_valid_index = get_predictions_from_rule(single_rule, test_df, sequence=sequence)
        valid_indexes.append(first_valid_index)
        prediction_list.append(sub_df)

    first_predictions = prediction_list[0]
    for i in range(1, len(prediction_list)):
        first_predictions["predictions"] = first_predictions["predictions"] | prediction_list[i]["predictions"]
    return first_predictions, min(valid_indexes)


def get_unique_fitness_rules(list_of_rules):
    fitness_list = [round(list_of_rules[0]["fitness"], 4)]
    unique_fitness_rules =  [list_of_rules[0]]
    for i in range(1, len(list_of_rules)):
        if round(list_of_rules[i]["fitness"], 4) not in fitness_list:
            fitness_list.append(round(list_of_rules[i]["fitness"], 4))
            unique_fitness_rules.append(list_of_rules[i])
    return unique_fitness_rules



def get_eval_dict(rules, kind, model_index, key, df, unique=False, sequence=False):
    if "uniq" in str(model_index) or unique == True:
        use_rules = get_unique_fitness_rules(rules)
    else:
        use_rules = rules
    if "avg" in str(model_index):
        ensemble_type = "avg"
    else:
        ensemble_type = "or"
    if kind == "rule":
        rule = use_rules[model_index]
        predict_df, first_valid_index = get_predictions_from_rule(rule, df, sequence=sequence)
    elif kind == "ensemble":
        if ensemble_type == "or":
            predict_df, first_valid_index = ensemble_learn_or(use_rules, df, sequence=sequence)
        else:
            predict_df, first_valid_index = ensemble_learn(use_rules, df, sequence=sequence)
    eval_dict = evaluate_prediction_model(predict_df, key, model_index=model_index, first_valid_index=first_valid_index)
    return eval_dict


def get_combo_eval_dict(eval_dict_list):
    sub_eval_dict = get_empty_eval_dict()
    empty_dict = {}
    #For each dictionary
    for sub_dict in eval_dict_list:
        #Append its value to this dictionary's list 
        for key, value in sub_dict.items():
            sub_eval_dict[key].append(value)
    #Rule index should be exactly the same 
    empty_dict["Rule Index"] =sub_eval_dict["Rule Index"][0]
    #Get mean of 
    empty_dict["Accuracy"] = sum(sub_eval_dict["Accuracy"])/len(sub_eval_dict["Accuracy"])
    sum_list = ["True_Negatives", "False_Positives", "False_Negatives", "True_Positives"]
    for sum_name in sum_list:
        empty_dict[sum_name] = sum(sub_eval_dict[sum_name])
    avg_list = ["Precision", "Recall", "F1 Score"]
    for avg_name in avg_list:
        empty_dict[avg_name] = sum(sub_eval_dict[avg_name])/len(sub_eval_dict[avg_name])
    return empty_dict



def complete_eval_top_rules(filepath_start, key, df, sequence=False):
    filename = f"{filepath_start}top_rules.json"
    if not os.path.exists(filepath_start):
        os.makedirs(filepath_start)
    rules_list = load_rules(filename)
    ensemble_indexes = ["ensemble_avg", "ensemble_uniq_avg", "ensemble_or", "ensemble_uniq_or"]
    model_indexes = [*range(0, len(rules_list))]
    both_indexes = model_indexes + ensemble_indexes

    eval_dict_list = []
    for single_index in both_indexes:
        if "ensemble" in str(single_index):
            kind = "ensemble"
        else:
            kind = "rule"
        if isinstance(df, list):
            sub_eval_dict_list = []
            for sub_df in df:
                sub_eval_dict = get_eval_dict(rules_list, kind, single_index, key, sub_df, sequence=sequence)
                sub_eval_dict_list.append(sub_eval_dict)
            eval_dict = get_combo_eval_dict(sub_eval_dict_list)
        else:
            eval_dict = get_eval_dict(rules_list, kind, single_index, key, df, sequence=sequence)
        eval_dict_list.append(eval_dict)
        #Change here 

    eval_df = pd.DataFrame(eval_dict_list)
    save_name = f"{filepath_start}rule_predictor_evaluation.csv"
    eval_df.to_csv(save_name)

#print(eval_dict)
#complete_eval_top_rules("generated_files/None/", "frost")



