import pandas as pd 
import math 
import ga_population
import ga_predictor 

#Help from here: https://www.geeksforgeeks.org/python-intersection-two-lists/


def return_default_parameter_dict():
    parameter_dict = {
        "mutation_rate": 20,
        "mutation_amount": 20,
        "range_restriction": 50,
        "index_key": "Date_datetime",
        "add_subtract_percent": 30,
        "change_percent": 70,
        "max_mutation_tries": 10,
        "population_size": 20, 
        "top_rules": 3,
        "generations": 3,
        "tournament_size": 15,
        "dominance": False
    }
    return parameter_dict.copy()


def return_default_consequent_dict():
    consequent_dict = {
        "name": "frost",
        "type": "boolean",
        "upper_bound": 1,
        "lower_bound": 1
    }
    return consequent_dict.copy()

def return_default_key():
    key = "frost" 
    return key 

def return_default_list_features_dict():
    list_features_dict = {
        "Air_TempC_Avg": {
            "name": "Air_TempC_Avg",
            "type": "continuous"
        },
        "Air_TempC_Max": {
            "name": "Air_TempC_Max",
            "type": "continuous"
        },
        "Air_TempC_Min": {
            "name": "Air_TempC_Min",
            "type": "continuous"
        },
        "Relative_Humidity_Avg": {
            "name": "Relative_Humidity_Avg",
            "type": "continuous"
        },
        "Relative_Humidity_Max": {
            "name": "Relative_Humidity_Max",
            "type": "continuous"
        },
        "Relative_Humidity_Min": {
            "name": "Relative_Humidity_Min",
            "type": "continuous"
        },
        "Ppt_mm_Tot": {
            "name": "Ppt_mm_Tot",
            "type": "continuous"
        },
        "WS_ms_300cm_Avg": {
            "name": "WS_ms_300cm_Avg",
            "type": "continuous"
        },
        "WS_ms_300cm_Max": {
            "name": "WS_ms_300cm_Max",
            "type": "continuous"
        },
        "WS_ms_150cm_Avg": {
            "name": "WS_ms_150cm_Avg",
            "type": "continuous"
        },
        "WS_ms_150cm_Max": {
            "name": "WS_ms_150cm_Max",
            "type": "continuous"
        },
        "WS_ms_75cm_Avg": {
            "name": "WS_ms_75cm_Avg",
            "type": "continuous"
        },
        "WS_ms_75cm_Max": {
            "name": "WS_ms_75cm_Max",
            "type": "continuous"
        },
        "WinDir_mean_Resultant": {
            "name": "WinDir_mean_Resultant",
            "type": "continuous"
        },
        "WinDir_Std_Dev": {
            "name": "WinDir_Std_Dev",
            "type": "continuous"
        }
    }
    return list_features_dict.copy()

def create_feature_dict(list_of_features, key, index_key):
    return_dict = {}
    for feature in list_of_features:
        if feature != key and feature != index_key: 
            sub_dict = {
                "name": feature,
                "type": "continuous"
            }
            return_dict[feature] = sub_dict
    return return_dict

#Merge the dataframes together, prefixing columns with the dataset they came from 
def create_merged_df(specific_site, sites, parameter_dict, key):
    columns_list = []
    default_features_list = return_default_list_features_dict()
    merge_index = parameter_dict["index_key"]
    features_to_use = list(default_features_list.keys())
    full_df = pd.DataFrame()
    for site in sites:
        df_path = f"frost_csvs/{site}.csv"
        df = pd.read_csv(df_path)
        if site == specific_site:
            keep_column = df[key]
        
        initial_cols = list(df.columns)
        features_to_use = list(set(features_to_use) & set(initial_cols))

        prefix_df = df[features_to_use]
        prefix_df = prefix_df.add_prefix(site)
        if site == specific_site:
            keep_column = df[key]
            prefix_df[key]=keep_column
        index_col = df[merge_index]
        prefix_df[merge_index] = index_col
        #Add these to the columns we are using
        columns_list = columns_list + (list(prefix_df.columns))
        if full_df.empty:
            full_df = prefix_df
        else:
            full_df = pd.merge(full_df, prefix_df, on=merge_index)
    list_features_dict = create_feature_dict(columns_list, key, merge_index)
    return full_df, list_features_dict
   



def run_experiments(phase_name, default_parameter_dict, name, sites, key=None, all_data=False, sequence=False):
    default_dict = return_default_parameter_dict()
    default_dict.update(default_parameter_dict)
    if key == None:
        key = return_default_key()
    consequent_dict = return_default_consequent_dict()
    for site in sites:
        full_name = name + f"_{site}"
        #Load in the info
        if all_data:
            list_features_dict = return_default_list_features_dict()
            df, list_features_dict = create_merged_df(site, sites, default_dict, key)
        else:
            df_path = f"frost_csvs/{site}.csv"
            df = pd.read_csv(df_path)
            list_features_dict = return_default_list_features_dict()

        #print(list_features_dict)
        #Split into training and test df 
        num_rows = len(df.index)
        #0.1 - 10 percent training set - kind of a magic number 
        split_index = num_rows - math.ceil(num_rows*0.2)
        train_df = df.iloc[:split_index, :]
        test_df = df.iloc[split_index:, :]
        test_df = test_df.reset_index()
        #print(list(test_df.columns))
        #CHANGE BACK 
        #Run the experiment
        pop = ga_population.population(default_dict, consequent_dict, list_features_dict, key, train_df)
        pop.run_experiment(name=full_name)
        #Eval - For each top rule and for the ensemble classifiers
        filename = f"generated_files/{full_name}/"
        ga_predictor.complete_eval_top_rules(filename, key, test_df, sequence=sequence)
        print("Finished ", site)




