import json
import pandas as pd 
import random 
import math 
import copy 
#from copy import deepcopy
import ga_rule
import os 
import numpy as np 
#d = deepcopy
#https://stackoverflow.com/questions/5326112/how-to-round-each-item-in-a-list-of-floats-to-2-decimal-places 




################################################# POPULATION CLASS ###################################################################
#How many top rules to hold? 
#10% hyperparameter of number of rules to hold in top 
#See first if we can init these rules, then worry about scoring them and making new populations 
class population:
    def __init__(self, default_parameter_dict, consequent_dict, feature_dict, df):
        #Passes parameters
        #Magic number for now 
        self.round_num = 2
        self.df = df 
        self.df_columns = self.get_df_columns(df)
        self.default_parameter_dict = default_parameter_dict.copy()
        self.consequent_dict = consequent_dict.copy()
        self.key = self.consequent_dict.get("name", None)
        for item in list(feature_dict.keys()):
            if item not in list(self.df_columns):
                feature_dict.pop(item) 
        self.features_dict = self.calc_parameters(feature_dict, self.default_parameter_dict, self.df, self.key)
         
        self.consequent_support, self.num_consequent, self.consequent_indexes = self.calc_consequent_support(self.consequent_dict, self.df)
        self.mutation_rate = self.default_parameter_dict['mutation_rate']
        self.population_size = self.default_parameter_dict["population_size"]
        self.num_top_rules = self.default_parameter_dict["top_rules"]
        self.generations = self.default_parameter_dict["generations"]
        self.tournament_size = self.default_parameter_dict["tournament_size"]
        self.dominance = self.default_parameter_dict["dominance"]
        if "diversify_top_rules" in list(self.default_parameter_dict.keys()):
            self.diversify_top_rules = self.default_parameter_dict["diversify_top_rules"]
        else:
            self.diversify_top_rules = False
        if "reseed_from_best" in list(self.default_parameter_dict.keys()):
            self.reseed_from_best = self.default_parameter_dict["reseed_from_best"]
        else:
            self.reseed_from_best = False
        
        self.mutation_number = math.ceil(self.population_size*(self.mutation_rate/100))
        
        #List of rules 
        self.rules_pop = self.init_rules_pop()
        self.top_rules = []

        #Dominance Dict 
        self.dominance_dict = {}
        self.dominance_fitness_dict = {}
        
    def get_df_columns(self, df):
        if isinstance(df, list):
            cols = []
            for sub_df in df:
                cols = cols + list(sub_df.columns)
            columns = [*set(cols)]
        else:
            columns = list(df.columns)
        return columns 


    def init_rules_pop(self):
        rules_pop = []
        for i in range(0, self.population_size):
            new_rule = ga_rule.rule(self.default_parameter_dict, self.features_dict, self.consequent_dict, self.consequent_support, self.num_consequent, self.consequent_indexes, self.df)
            #new_rule.random_init()
            rules_pop.append(new_rule)
        return rules_pop

    def calc_consequent_support(self, consequent_dict, df):
        param_name = consequent_dict['name']
        lower_bound = consequent_dict['lower_bound']
        upper_bound = consequent_dict['upper_bound']
        #Change also here! 
        query = f'`{param_name}` >= {lower_bound} & `{param_name}` <= {upper_bound}'

        if isinstance(df, list):
            num_consequent = 0
            index_list = []
            df_possible = 0
            for actual_df in df:
                sub_df = actual_df.eval(query)
                num_consequent += sub_df.sum()
                df_possible += len(actual_df.index)
                indexes = sub_df[sub_df].index
                index_list.append(np.array(indexes.tolist()))
        else:
            sub_df = df.eval(query)
            num_consequent = sub_df.sum()
            df_possible = len(df.index)
            indexes = sub_df[sub_df].index
            index_list = np.array(indexes.tolist())

        consequent_support = num_consequent/df_possible
        return consequent_support, num_consequent, index_list


    def default_features(self, param_name, feature, default_parameter_dict, defaults_list):
        for item in defaults_list:
            if item not in list(feature.keys()):
                if item == "name":
                    feature["name"] = param_name
                else:
                    if item in list(default_parameter_dict.keys()):
                        feature[item] = default_parameter_dict[item]
                    else:
                        feature[item] = False
        if feature["sequence"] == False:
            feature["sequence_limit"] = False
            feature["sequence_penalty"] = False
        return feature 


    def make_df_list(self, df_list, name):
        final_list = []
        for sub_df in df_list:
            if name in sub_df.columns:
                final_list = final_list + sub_df[name].dropna().values.tolist()
        return final_list


    def calculated_params(self, feature, df):
        if isinstance(df, list):
            df_list = self.make_df_list(df, feature["name"])
        else:
            df_list = df[feature["name"]].dropna().values.tolist()
        #Get max and min value for feature if they were not provided
        if "lower_bound" not in list(feature.keys()):
                feature["lower_bound"] = min(df_list)
        if "upper_bound" not in list(feature.keys()):
                feature["upper_bound"] = max(df_list)  
        #Mean and Stdev 
        if feature["type"] == "continuous" or feature["type"] == "nominal":
            feature["mean"] = np.mean(df_list)
            feature["stdev"] = np.std(df_list)
        return feature


    def calc_parameters(self, feature_dict, default_parameter_dict, df, key):
        #For each 
        defaults_list = ["name", "mutation_amount", "range_restriction", "range_penalty", "max_mutation_tries", "sequence", "sequence_limit", "sequence_penalty", "sequence_offset"]
        for item in list(feature_dict.keys()):
            feature = feature_dict[item]
            #Load in defaults that aren't already present 
            feature = self.default_features(item, feature, default_parameter_dict, defaults_list)
            #Calculated Stuff!! 
            feature = self.calculated_params(feature, df)
        return feature_dict

    
    def curb_diversify_top_rules(self):
        unique_rule_strings = {}
        #Get highest fitness for given parameter set 
        for rule in self.top_rules:
            rule_string = str(sorted(list(rule.get_rule_dict().keys())))
            if rule_string not in unique_rule_strings.keys():
                unique_rule_strings[rule_string] = rule.get_fitness()
            else:
                fitness = rule.get_fitness()
                if fitness > unique_rule_strings[rule_string]:
                    unique_rule_strings[rule_string] = fitness
        keep_list = []
        for rule in self.top_rules:
            rule_string = str(sorted(list(rule.get_rule_dict().keys())))
            if rule.get_fitness() == unique_rule_strings[rule_string]:
                keep_list.append(rule)
        self.top_rules = keep_list

            

    def update_top_rules(self):
        #get the top rules in the generation
        self.rules_pop.sort(reverse=True)
        #Get the top keep rules from this population:
        self.pop_top_rules = copy.deepcopy(self.rules_pop[:self.num_top_rules])
        new_pop_top_rules = []
        if self.top_rules == []:
            self.top_rules = copy.deepcopy(self.pop_top_rules)
        #SO UGLY - CHECK 
        else:
            for rule in self.pop_top_rules:
                #Assume not same
                same = False
                for other_rule in self.top_rules:
                    active_params = rule.get_active_parameters()
                    other_active_params = other_rule.get_active_parameters()
                    bounds = rule.get_bounds_list()
                    bounds = [round(item, self.round_num) for item in bounds]
                    other_bounds = other_rule.get_bounds_list()
                    other_bounds = [round(item, self.round_num) for item in other_bounds]
                    if active_params == other_active_params and bounds == other_bounds:
                        same = True
                    #CHANGE here - not sure if good or not. 
                    elif active_params == other_active_params and rule.get_fitness() < other_rule.get_fitness():
                        same = True
                if same == False:
                    new_pop_top_rules.append(rule)
                    
            temp_top_list = self.top_rules + new_pop_top_rules
            temp_top_list.sort(reverse=True)
            self.top_rules = copy.deepcopy(temp_top_list[:self.num_top_rules])
            #print(len(self.top_rules))

    def mutate_population(self):
        mutate_rules = random.sample(self.rules_pop, self.mutation_number)        
        for rule in mutate_rules:
            #print(rule.print_self())
            rule.mutate(self.df)

    def update_dominance_dict(self):
        #Dominance rules need to have lower fitness to be killed, I think 
        for rule in self.rules_pop:
            rule_dict = rule.get_rule_dict()
            #Make its parameters a string - sort alpha so always same
            rule_string = str(sorted(list(rule_dict.keys())))
            #If we don't have an entry for this, make one
            if rule_string not in list(self.dominance_dict.keys()):
                self.dominance_dict[rule_string] = copy.deepcopy(rule_dict)
                self.dominance_fitness_dict[rule_string] = rule.get_fitness()
            #Otherwise:
            else:
                compare_rule_dict = self.dominance_dict[rule_string]
                dominated = True
                for param in list(rule_dict.keys()):
                    #But if it is NOT dominated on anything:
                    if round(rule_dict[param].upper_bound, self.round_num) > round(compare_rule_dict[param].upper_bound, self.round_num) and round(rule_dict[param].lower_bound, self.round_num) < round(compare_rule_dict[param].lower_bound, self.round_num):
                        dominated = False
                #If its not dominated, we put the non dominated rule into the dict at these parameters 
                if dominated == False:
                    self.dominance_dict[rule_string] = copy.deepcopy(rule_dict)
                    self.dominance_fitness_dict[rule_string] = rule.get_fitness()


    def kill_dominated(self):
        new_rules_pop_list = []
        for rule in self.rules_pop:
            rule_dict = rule.get_rule_dict()
            #Make its parameters a string - sort alpha so always same
            rule_string = str(sorted(list(rule_dict.keys())))
            #print(rule_string)
            compare_rule_dict = self.dominance_dict[rule_string]
            #Assume dominated
            dominated = True
            for param in list(rule_dict.keys()):
                #But if it is NOT dominated on anything:
                #CHANGE HERE - potentially a VERY bad one. 
                if rule_dict[param].curr_upper_bound > compare_rule_dict[param].curr_upper_bound and rule_dict[param].curr_lower_bound < compare_rule_dict[param].curr_lower_bound:
                    dominated = False
            if dominated == False:
                self.dominance_dict[rule_string] = copy.deepcopy(rule_dict)
            #Add it if it's fitness is higher!
            if dominated:
                #Only keep if it has a higher fitness
                #Another potentially bad change! 
                if rule.fitness > self.dominance_fitness_dict[rule_string]:
                    #CHANGE HERE - not sure if good or not! 
                    self.dominance_dict[rule_string] = copy.deepcopy(rule.get_rule_dict())
                    self.dominance_fitness_dict[rule_string] = rule.fitness
                    new_rules_pop_list.append(rule)
                else:
                    #print("Killing ")
                    #rule.elegant_print()
                    pass 
            else:
                new_rules_pop_list.append(rule)
        self.rules_pop = new_rules_pop_list

    def tournament_competition(self): 
        #Randomly pick 4 of the rules from the rule pool
        competitors = random.sample(self.rules_pop, self.tournament_size)
        fittest = competitors[0]
        fittest_fitness = competitors[0].get_fitness()
        for i in range(1, self.tournament_size):
            curr_fitness = competitors[i].get_fitness()
            if curr_fitness > fittest_fitness:
                fittest_fitness = curr_fitness
                fittest = competitors[i]
        return copy.deepcopy(fittest)

    def tournament_selection(self):
        new_pop = []
        for i in range(0, self.population_size):
            offspring = self.tournament_competition()
            new_pop.append(offspring)
        self.rules_pop = new_pop 


    #NOTE: YOU MIGHT WANT TO DELETE 0 FITNESS INDIVIDUALS
    def run_generation(self):
        #Update dominance dict and Kill dominated rules
        #Take another look at this - might incorporate into fitness 
        if self.dominance:
            self.update_dominance_dict()
            self.kill_dominated()
        #Kill lowest 20% of rules - MAGIC NUMBER ALERT 
        else:  
            self.rules_pop.sort()
            self.rules_pop = self.rules_pop[math.ceil(len(self.rules_pop)*.20):]


        #CHANGE HERE
        #This kills all but the best with the same parameters. Which might be an awful idea. 
        if self.diversify_top_rules:
            self.curb_diversify_top_rules()

        #Update the top rules
        self.update_top_rules()

        #Replace dead population members
        num_replacements = self.population_size - len(self.rules_pop)
        for i in range(0, num_replacements):
            #How will we make the next seed?
            #Magic NUMBER ALERT - CHECK 
            if self.reseed_from_best:
                seed = random.choices(["best", "new"], weights=[10, 90], k=1)[0]
                if seed == "best" and len(self.top_rules) > 0:
                    new_rule = copy.deepcopy(random.choice(self.top_rules))
                else:
                    new_rule = ga_rule.rule(self.default_parameter_dict, self.features_dict, self.consequent_dict, self.consequent_support, self.num_consequent, self.consequent_indexes, self.df)
            else:
                new_rule = ga_rule.rule(self.default_parameter_dict, self.features_dict, self.consequent_dict, self.consequent_support, self.num_consequent, self.consequent_indexes, self.df)
            self.rules_pop.append(new_rule)
        #Create the next generation
        self.tournament_selection()
        #Mutate percentage of population
        self.mutate_population() 


    def save_top_rules(self, name=None):
        list_of_rules = []
        for rule in self.top_rules:
            list_of_rules.append(rule.get_rule_dict_all_numeric())
        rule_save = json.dumps(list_of_rules, indent=4)
        start_string = f"data/output_data/{name}/"
        if not os.path.exists(start_string):
            os.makedirs(start_string)
        save_string = f"{start_string}top_rules.json"
        with open(save_string, "w") as f:
            f.write(rule_save) 


    def save_all_rules(self, name=None):
        list_of_rules = []
        for rule in self.rules_pop:
            list_of_rules.append(rule.get_rule_dict_all_numeric())
        rule_save = json.dumps(list_of_rules, indent=4)
        start_string = f"data/output_data/{name}/"
        if not os.path.exists(start_string):
            os.makedirs(start_string)
        save_string = f"{start_string}all_rules.json"
        with open(save_string, "w") as f:
            f.write(rule_save) 


    def run_experiment(self, status=False, name=None):
        #Run the generations 
        for i in range(0, self.generations):
            if status:
                print(f" Generation {i}")
            self.run_generation()
        #Save the rules 
        self.save_top_rules(name=name)
        self.save_all_rules(name=name)

    def print_self(self):
        print(f"Pop size: ", self.population_size)
        print(f"Number of top rules to retain: ", self.num_top_rules)
        
    def print_rules(self):
        print("Rules: ")
        for item in self.rules_pop:
            item.print_self()

    def print_rules_and_fitness(self):
        print("Rules: ")
        for item in self.rules_pop:
            item.print_self()
            item.print_fitness_metrics()
            print()

    
    def print_top_rules_and_fitness(self):
        print("Global top rules metrics")
        for rule in self.top_rules:
            rule.elegant_print()
            rule.print_fitness_metrics()
            print()

    def print_dominance_dict(self):
        for item in list(self.dominance_dict.keys()):
            print(self.dominance_dict[item].keys())

