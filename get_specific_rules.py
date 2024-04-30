import json
#Data-Specific Helpers for this particular dataset 
import data.input_data.lgwf_specific_data as lgwf
import qarm_genetic_algorithm 
#lists
import importlib 


def load_rules(filename):
    with open(filename) as f:
        rules_list = json.load(f)
    return rules_list
    #print(json.dumps(rules_list, indent=4))



def get_rule(run_name, indexes):
    filename = f"data/output_data/{run_name}/top_rules.json"
    rules_list = load_rules(filename)

    for index in indexes:
        print(json.dumps(rules_list[index], indent=4))

run_name = "run_56"
indexes = [7,9,12,19]


get_rule(run_name, indexes)


