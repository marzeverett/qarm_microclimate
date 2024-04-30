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


run_name = "run_45"
indexes = [19, 29]

filename = f"data/output_data/{run_name}/top_rules.json"
rules_list = load_rules(filename)

for index in indexes:
    print(json.dumps(rules_list[index], indent=4))