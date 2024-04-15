import pandas as pd 
import math 
import ga_population
import ga_predictor 

#Help from here: 
#https://www.programiz.com/python-programming/datetime/current-time 

def run_experiments(experiment_name, parameter_dict, consequent_dict, feature_dict, train_df, test_df):
        pop = ga_population.population(parameter_dict, consequent_dict, feature_dict, train_df)
        pop.run_experiment(name=experiment_name)
        #Eval - For each top rule
        key = consequent_dict.get("name", None)
        sequence = parameter_dict.get("sequence", False)
        filename = f"data/output_data/{experiment_name}/"
        ga_predictor.complete_eval_top_rules(filename, key, test_df, sequence=sequence)
        print(f"Finished {experiment_name}")

def run_just_predictor(experiment_name, parameter_dict, consequent_dict, feature_dict, train_df, test_df):
        #Eval - For each top rule
        key = consequent_dict.get("name", None)
        sequence = parameter_dict.get("sequence", False)
        filename = f"data/output_data/{experiment_name}/"
        ga_predictor.complete_eval_top_rules(filename, key, test_df, sequence=sequence)
        print(f"Finished predicting {experiment_name}")



