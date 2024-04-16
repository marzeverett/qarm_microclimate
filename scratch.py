import json
#Data-Specific Helpers for this particular dataset 
import data.input_data.lgwf_specific_data as lgwf
import qarm_genetic_algorithm 
#lists
import importlib 




param_dict =  {
        "mutation_rate": 20,
        "mutation_amount": 70,
        "range_restriction": False,
        "range_penalty": False,
        "initial_rule_limit": 4,
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "add_subtract_percent": 30,
        "change_percent": 70,
        "max_mutation_tries": 5,
        "population_size": 150, 
        "top_rules": 10,
        "generations": 20,
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

name = "testing_1"

#Get the feature_dict, consequent dict, and df's
feature_dict = lgwf.get_feature_dictionary(param_dict)
consequent_dict = lgwf.get_consequent_dict(param_dict)
train_df, test_df = lgwf.get_train_and_test_df(param_dict)

train_cols = train_df.columns.tolist()
test_cols = test_df.columns.tolist()

all_cols = train_cols.copy()

#https://www.jitsejan.com/find-and-delete-empty-columns-pandas-dataframe
empty_train_cols = [col for col in train_df.columns if train_df[col].isnull().all()]
empty_test_cols = [col for col in test_df.columns if test_df[col].isnull().all()]

unusable_cols = empty_train_cols + empty_test_cols
unusable_cols = list(set(unusable_cols))

#https://stackoverflow.com/questions/4211209/remove-all-the-elements-that-occur-in-one-list-from-another 
#all_cols - unusable_cols 
usable_cols = [x for x in all_cols if x not in unusable_cols]

print("UNUSABLE COLS")
print(unusable_cols)

print("USABLE COLS")
print(usable_cols)

print(train_df.head())
print(test_df.head())

# # Drop these columns from the dataframe
# df.drop(empty_cols,
#         axis=1,
#         inplace=True)