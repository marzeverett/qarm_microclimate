import json
#Data-Specific Helpers for this particular dataset 
import data.input_data.lgwf_specific_data as lgwf
import qarm_genetic_algorithm 
#lists
import importlib 

basic_param_dict =  {
        "mutation_rate": 50,
        "mutation_amount": 100,
        "range_restriction": False,
        "range_penalty": False,
        "initial_rule_limit": 4,
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "add_subtract_percent": 30,
        "change_percent": 70,
        "max_mutation_tries": 5,
        "population_size": 150, 
        "top_rules": 20,
        "generations": 100,
        "tournament_size": 4,
        "dominance": True,
        "sequence": True,
        "sequence_limit": 120,
        "sequence_penalty": False,
        "sequence_offset": 13,
        "diversify_top_rules": True,
        "reseed_from_best": False,
        "sequence_antecedent_heuristic": False,
        "fitness_function_index": 13,
        "sequence_penalty_index": -1,
        "range_penalty_index": -1
    }

def run_rule_predictor(experiment_name, rules_list, key):
    #Get the feature_dict, consequent dict, and df's
    train_df, test_df = lgwf.get_train_and_test_df(basic_param_dict)
    #Run the experiment
    qarm_genetic_algorithm.run_rule_predictor(experiment_name, rules_list, key, test_df, True)

#Take a list of rules in a file name 
#Get the predictions graphed against actual for all rules
#Save the predictions
#Save the stats 

#Think you need the consequent dict. (Name and sequence or not. You also need the test df. )
key = "delta_frost_events"
#rule_list_name = "high_recall_rules"
rule_list_name = "sample_8"
rules_list_module = importlib.import_module(f"rules.{rule_list_name}", package=None)
rules_list = rules_list_module.rules
#Run all the experiments in the file through prediction
run_rule_predictor(rule_list_name, rules_list, key)

