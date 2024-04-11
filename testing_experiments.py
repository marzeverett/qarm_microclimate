import json
#Data-Specific Helpers for this particular dataset 
import data.input_data.lgwf_specific_data as lgwf
import qarm_genetic_algorithm 

param_dict = {
    "mutation_rate": 20,
    "mutation_amount": 20,
    "range_restriction": False,
    "range_penalty": True,
    "initial_rule_limit": 2,
    "index_key": "time_steps",
    "consequent_key": "all_frost_events",
    "add_subtract_percent": 30,
    "change_percent": 70,
    "max_mutation_tries": 5,
    "population_size": 150, 
    "top_rules": 10,
    #"generations": 150,
    "generations": 30,
    "tournament_size": 4,
    "dominance": True,
    "sequence": True,
    "sequence_limit": 36,
    "sequence_penalty": True,
    "sequence_offset": 12,
    "diversify_top_rules": True,
    "reseed_from_best": True,
    "sequence_antecedent_heuristic": False,
    "fitness_function_index": 7,
    "sequence_penalty_index": 4,
    "range_penalty_index": 0
}


#Get the feature_dict - 

feature_dict = lgwf.get_feature_dictionary(param_dict)
consequent_dict = lgwf.get_consequent_dict(param_dict)
train_df, test_df = lgwf.get_train_and_test_df(param_dict)

name = "test_experiment_3"

qarm_genetic_algorithm.run_experiments(name, param_dict, consequent_dict, feature_dict, train_df, test_df)



