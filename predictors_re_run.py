import json
#Data-Specific Helpers for this particular dataset 
import data.input_data.lgwf_specific_data as lgwf
import qarm_genetic_algorithm 
#lists
import importlib 

def run_lgwf_experiment(name, param_dict):
    #Get the feature_dict, consequent dict, and df's
    feature_dict = lgwf.get_feature_dictionary(param_dict)
    consequent_dict = lgwf.get_consequent_dict(param_dict)
    train_df, test_df = lgwf.get_train_and_test_df(param_dict)

    #Run the experiment
    qarm_genetic_algorithm.run_experiments(name, param_dict, consequent_dict, feature_dict, train_df, test_df)

    #Save parameter dict 
    json_params = json.dumps(param_dict)
    with open(f"data/output_data/{name}/parameter_dict.json", 'w') as outfile:
        json.dump(json_params, outfile)

def run_lgwf_predictor(name, param_dict):
    #Get the feature_dict, consequent dict, and df's
    feature_dict = lgwf.get_feature_dictionary(param_dict)
    consequent_dict = lgwf.get_consequent_dict(param_dict)
    train_df, test_df = lgwf.get_train_and_test_df(param_dict)
    #Run the experiment
    qarm_genetic_algorithm.run_just_predictor(name, param_dict, consequent_dict, feature_dict, train_df, test_df)




experiment_list_file_name = "run_1"
#experiment_list_file_name = "experiments_list"

experiments_list = importlib.import_module(f"experiment_parameters.{experiment_list_file_name}", package=None)

#Run all the experiments in the file through prediction
for experiment_name, experiment_parameters in experiments_list.experiments.items():
    run_lgwf_predictor(experiment_name, experiment_parameters)


