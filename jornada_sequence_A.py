import json
import run_experiments 

param_dict_1 = {
    "mutation_rate": 20,
    "mutation_amount": 20,
    "range_restriction": False,
    "range_penalty": True,
    "initial_rule_limit": 2,
    "index_key": "Date_datetime",
    "add_subtract_percent": 30,
    "change_percent": 70,
    "max_mutation_tries": 5,
    "population_size": 150, 
    "top_rules": 10,
    "generations": 150,
    "tournament_size": 4,
    "dominance": True,
    "sequence": True,
    "sequence_limit": 12,
    "sequence_penalty": True,
    "diversify_top_rules": True,
    "reseed_from_best": True,
    "sequence_antecedent_heuristic": False,
    "fitness_function_index": 1,
    "sequence_penalty_index": 2,
    "range_penalty_index": 0
}

param_dict_2 = {
    "mutation_rate": 20,
    "mutation_amount": 20,
    "range_restriction": False,
    "range_penalty": True,
    "initial_rule_limit": 2,
    "index_key": "Date_datetime",
    "add_subtract_percent": 30,
    "change_percent": 70,
    "max_mutation_tries": 5,
    "population_size": 150, 
    "top_rules": 10,
    "generations": 150,
    "tournament_size": 4,
    "dominance": True,
    "sequence": True,
    "sequence_limit": 12,
    "sequence_penalty": True,
    "diversify_top_rules": True,
    "reseed_from_best": True,
    "sequence_antecedent_heuristic": False,
    "fitness_function_index": 1,
    "sequence_penalty_index": 3,
    "range_penalty_index": 0
}

param_dict_3 = {
    "mutation_rate": 20,
    "mutation_amount": 20,
    "range_restriction": False,
    "range_penalty": True,
    "initial_rule_limit": 2,
    "index_key": "Date_datetime",
    "add_subtract_percent": 30,
    "change_percent": 70,
    "max_mutation_tries": 5,
    "population_size": 150, 
    "top_rules": 10,
    "generations": 150,
    "tournament_size": 4,
    "dominance": True,
    "sequence": True,
    "sequence_limit": 12,
    "sequence_penalty": True,
    "diversify_top_rules": True,
    "reseed_from_best": True,
    "sequence_antecedent_heuristic": False,
    "fitness_function_index": 2,
    "sequence_penalty_index": 2,
    "range_penalty_index": 0
}

param_dict_4 = {
    "mutation_rate": 20,
    "mutation_amount": 20,
    "range_restriction": False,
    "range_penalty": True,
    "initial_rule_limit": 2,
    "index_key": "Date_datetime",
    "add_subtract_percent": 30,
    "change_percent": 70,
    "max_mutation_tries": 5,
    "population_size": 150, 
    "top_rules": 10,
    "generations": 150,
    "tournament_size": 4,
    "dominance": True,
    "sequence": True,
    "sequence_limit": 12,
    "sequence_penalty": True,
    "diversify_top_rules": True,
    "reseed_from_best": True,
    "sequence_antecedent_heuristic": False,
    "fitness_function_index": 2,
    "sequence_penalty_index": 3,
    "range_penalty_index": 0
}


npp_named_sites = ['npp_c_cali', 'npp_c_grav', 'npp_c_sand', 'npp_g_basn', 'npp_g_ibpe', 'npp_g_summ', 'npp_m_nort', 'npp_m_rabb', 'npp_m_well', 'npp_p_coll', 'npp_p_smal', 'npp_p_tobo', 'npp_t_east', 'npp_t_tayl', 'npp_t_west']
runs = {
    "1": "1",
    "2": "2",
    "3": "3",
    }
params_dicts = {
    "1": param_dict_1,
    "2": param_dict_2,
    "3": param_dict_3,
    "4": param_dict_4
}
#NAME - {phase_name}_{parameter_index}_{Run}
phase_name = "Sequence_A"
key="frost"
sequence=True
all_data=False
for param_dict_index in list(params_dicts.keys()):
    for run_index in list(runs.keys()):
        name = f'{phase_name}_{param_dict_index}_{run_index}'
        run_experiments.run_experiments(phase_name, params_dicts[param_dict_index], name, npp_named_sites, key=key, all_data=all_data, sequence=True)
        print("Finished run...")

#Same as run 2, but with diversity measures (non re-seeding with best, and 
#diversifying the top rules.)

