import json
import pandas as pd 
import random 
import math 
import copy 
import ga_parameter
import numpy as np 


#Not easy but fairly straightforward, at least. 
#First, let's worry about init'ing and eval'ing. Then,
#Lets worry about the scoring. 

#The scoring is 90% of the work here. 

#This is probably the place where most of the optimizations will need to take place
#Look at this article for queries: https://saturncloud.io/blog/the-fastest-way-to-perform-complex-search-on-pandas-dataframe/ 
#https://stackoverflow.com/questions/41125909/find-elements-in-one-list-that-are-not-in-the-other
#Numpy Help: https://numpy.org/doc/stable/reference/generated/numpy.intersect1d.html 

#######################       RULE CLASS             #########################################
class rule:
    def __init__(self, default_parameter_dict, features_dict, consequent, consequent_support, num_consequent, consequent_indexes, df):
        self.features_dict = features_dict.copy()
        self.parameter_list = list(self.features_dict.keys())
        self.mutation_rate = default_parameter_dict["mutation_rate"]
        self.add_subtract_percent = default_parameter_dict['add_subtract_percent']
        self.change_percent = default_parameter_dict['change_percent']
        self.max_mutation_tries = default_parameter_dict["max_mutation_tries"]
        self.sequence = default_parameter_dict["sequence"]
        if "sequence_penalty" in list(default_parameter_dict.keys()):
            self.sequence_penalty = default_parameter_dict["sequence_penalty"]
        else:
            self.sequence_penalty = False
        if "range_penalty" in list(default_parameter_dict.keys()):
            self.range_penalty = default_parameter_dict["range_penalty"]
        else:
            self.range_penalty = False
        if "fitness_function_index" in list(default_parameter_dict.keys()):
            self.fitness_function_index = default_parameter_dict["fitness_function_index"]
        else:
            self.fitness_function_index = 0
        if "sequence_penalty_index" in list(default_parameter_dict.keys()):
            self.sequence_penalty_index = default_parameter_dict["sequence_penalty_index"]
        else:
            self.sequence_penalty_index = 0
        if "sequence_limit" in list(default_parameter_dict.keys()):
            self.sequence_limit = default_parameter_dict["sequence_limit"]
        else:
            self.sequence_limit = False

        if "range_penalty_index" in list(default_parameter_dict.keys()):
            self.range_penalty_index = default_parameter_dict["range_penalty_index"]
        else:
            self.range_penalty_index = 0
        if "initial_rule_limit" in list(default_parameter_dict.keys()):
            self.init_max_params = default_parameter_dict["initial_rule_limit"]
        else:
            self.init_max_params = math.ceil(0.6*len(self.parameter_list))

        if "sequence_antecedent_heuristic" in list(default_parameter_dict.keys()):
            self.sequence_antecedent_heuristic = default_parameter_dict["sequence_antecedent_heuristic"]
        else:
            self.sequence_antecedent_heuristic = False

        if "sequence_offest" in list(default_parameter_dict.keys()):
            self.sequence_offset = default_parameter_dict["offset"]
        else:
            self.sequence_offset = False


        self.consequent_dict = consequent
        self.consequent_support = consequent_support
        self.num_consequent = num_consequent
        self.consequent_indexes = consequent_indexes
        if isinstance(df, list):
            total_records = 0
            for sub_df in df:
                total_records += len(sub_df.index)
        else:
            total_records = len(df.index)
        self.total_records = total_records
        self.rule_dict = {}
        self.active_parameters = []
        self.last_mutation_type = None
        #CHECK: Magic Number Alert 
        self.max_init_tries = 5
        
        #Make sure we initialize the rule to something actually in the dataset 
        self.antecedent_support = 0
        init_initial = 0
        while self.antecedent_support <= 0.0 and init_initial <= self.max_init_tries:
            self.random_init()
            self.calc_antecedent_support(df)
            init_initial += 1
        self.calc_fitness(df)


    def random_init(self):
        self.rule_dict = {}
        self.active_parameters = []
        #One rule
        #Pick a number of parameters
        #num = random.uniform(0, self.init_max_params)
        #This is where we should maybe change - MARKED CHANGE HERE - eventually paramterize 

        num = random.uniform(1, self.init_max_params)

        working_list = self.parameter_list.copy()
        round_num = math.ceil(num)
        if round_num == 0:
            round_num = 1
        #For the number of parameters we decided to go with 
        for i in range(0, round_num):
            #Pick a parameter
            parameter_name = random.choice(working_list)
            #Pop that off the working list
            working_list.remove(parameter_name)
            #Init the parameter
            #Add it to the rule dict, indexed by its name 
            self.rule_dict[parameter_name] = ga_parameter.parameter(parameter_name, self.features_dict)
            self.active_parameters.append(parameter_name)
        

    def build_rule_antecedent_query(self):
        query_string = ''
        first = 1
        for item in list(self.rule_dict.keys()):
            lower, upper = self.rule_dict[item].return_bounds()
            if not first:
                query_string = query_string + ' & '
            query_string = query_string + f'{item} >= {lower} & {item} <= {upper}'
            first = 0
        self.antecedent_support_query = query_string
        return query_string

    def build_consequent_query(self):
        param_name = self.consequent_dict['name']
        lower_bound = self.consequent_dict['lower_bound']
        upper_bound = self.consequent_dict['upper_bound']
        query = f'{param_name} >= {lower_bound} & {param_name} <= {upper_bound}'
        self.consequent_support_query = query
        return query 


    def build_param_specific_query(self, param_name):
        parameter = self.rule_dict[param_name]
        lower, upper = parameter.return_bounds()
        #CHANGE IS HERE
        #query = f'{param_name} >= {lower} & {param_name} <= {upper}'
        query = f'`{param_name}` >= {lower} & `{param_name}` <= {upper}'
        return query  

    def get_indexes(self, param_name, df):
        query = self.build_param_specific_query(param_name)
        bool_df = df.eval(query)
        indexes = bool_df[bool_df].index
        index_list = indexes.tolist()
        return index_list

    def build_fulfilment_indexes(self, param_name, param_indexes, len_df):
        overall_list = []
        lower, upper = self.rule_dict[param_name].return_seq_bounds()
        adding_list = list(range(lower, upper+1))
        fulfilled_indexes = None
        first = True
        for add_val in adding_list:
            raw_indexes = np.array(param_indexes)
            added_indexes = raw_indexes + add_val
            if first:
                fulfilled_indexes = added_indexes
                first=False
            else:
                fulfilled_indexes = np.concatenate((fulfilled_indexes, added_indexes), axis=0)
        final = np.unique(fulfilled_indexes)
        final = final[final < len_df]
        final = final[final >= 0]
        return final

    def get_antecedent_indexes(self, df):
        final_indexes = None
        first = True
        for param_name in list(self.rule_dict.keys()):
            #Get the indexes where the parameters are between those values
            param_indexes = self.get_indexes(param_name, df)
            #Get the indexes that the parameters alone would fulfill as potential consequents
            fulfilled_indexes = self.build_fulfilment_indexes(param_name, param_indexes, len(df.index))
            #Return the intersection of these and the existing fulfilled parameters. Must be in all. 
            if first:
                final_indexes = fulfilled_indexes
                first=False
            else:
                final_indexes = np.intersect1d(final_indexes, fulfilled_indexes, assume_unique=True)
            #If it's ever the case the a new list doesn't have something in common with the current one:
            if final_indexes.size == 0:
                break 
                #numpy.intersect1d(ar1, ar2, assume_unique=False, return_indices=False)
        return final_indexes

    
    def get_indexes_and_applicability(self, df):
        final_indexes = self.get_antecedent_indexes(df)
        max_lower_bound = self.get_outlier_sequence_bounds("lower", "max")
        antecedent_applicable = len(df.index)-max_lower_bound
        return final_indexes, antecedent_applicable

    #Get the support of the antecedent for a sequence. 
    def calc_antecedent_support_sequence(self, df):
        if isinstance(df, list):
            sub_num_antecedent = 0
            sub_applicable = 0 
            antecedent_indexes = []
            for sub_df in df:
                sub_final_indexes, applicable = self.get_indexes_and_applicability(sub_df)
                sub_num_antecedent += sub_final_indexes.size
                sub_applicable += applicable
                antecedent_indexes.append(sub_final_indexes)
        else:
            sub_final_indexes, applicable = self.get_indexes_and_applicability(df)
            sub_num_antecedent = sub_final_indexes.size
            sub_applicable = applicable
            antecedent_indexes = sub_final_indexes

        self.num_antecedent = sub_num_antecedent
        self.antecedent_applicable = sub_applicable
        self.antecedent_support = self.num_antecedent/self.antecedent_applicable
        self.antecedent_indexes = antecedent_indexes


    def get_non_sequence_indexes_and_applicability(self, df):
        final_indexes = None
        first = True
        for param_name in list(self.rule_dict.keys()):
            #Get the indexes where the parameters are between those values
            param_indexes = self.get_indexes(param_name, df)
            param_indexes = np.array(param_indexes)
            #Get the indexes that the parameters alone would fulfill as potential consequents
            #Return the intersection of these and the existing fulfilled parameters. Must be in all. 
            if first:
                final_indexes = param_indexes
                first=False
            else:
                final_indexes = np.intersect1d(final_indexes, param_indexes, assume_unique=True)
            if final_indexes.size == 0:
                break 
        return final_indexes, len(df.index)

    def calc_antecedent_support_non_sequence(self, df):
        #Takes in itself and the dataframe, and calculates its support 
        #Get the indexes of the parameters
        if isinstance(df, list):
            total_indexes = 0
            total_applicable = 0
            antecedent_indexes = []
            for sub_df in df:
                sub_indexes, sub_total = self.get_non_sequence_indexes_and_applicability(sub_df)
                total_indexes += sub_indexes.size
                antecedent_indexes.append(antecedent_indexes)
                total_applicable += sub_total

        else:
            antecedent_indexes, total_applicable = self.get_non_sequence_indexes_and_applicability(df)
            total_indexes = antecedent_indexes.size 
        #Recalc 
        self.num_antecedent = total_indexes
        self.antecedent_support = total_indexes/total_applicable
        self.antecedent_indexes = antecedent_indexes
        self.antecedent_applicable = total_applicable


    def calc_antecedent_support(self, df):
        if self.sequence:
            self.calc_antecedent_support_sequence(df)
        else:
            self.calc_antecedent_support_non_sequence(df)

    
    def calc_overall_support_sequence(self, df):
        if isinstance(df, list):
            same_indexes =  []
            sub_num_whole_rule = 0
            for i in range(0, len(self.antecedent_indexes)):
                sub_same_indexes = np.intersect1d(self.antecedent_indexes[i], self.consequent_indexes[i], assume_unique=True)
                same_indexes.append(sub_same_indexes)
                sub_num_whole_rule += sub_same_indexes.size
        else:
            same_indexes = np.intersect1d(self.antecedent_indexes, self.consequent_indexes, assume_unique=True)
            sub_num_whole_rule = same_indexes.size
        
        self.num_whole_rule = sub_num_whole_rule
        self.whole_rule_indexes = same_indexes
        #print("Num whole rule ", self.num_whole_rule)
        self.support = self.num_whole_rule/self.antecedent_applicable


    #CHANGE IS HERE, CHECK 
    def calc_overall_support_non_sequence(self, df):
        # same_indexes = np.intersect1d(self.antecedent_indexes, self.consequent_indexes, assume_unique=True)
        # self.num_whole_rule = same_indexes.size
        # self.whole_rule_indexes = same_indexes
        # #print("Num whole rule ", self.num_whole_rule)
        # self.support = self.num_whole_rule/self.total_records

        if isinstance(df, list):
            same_indexes =  []
            sub_num_whole_rule = 0
            for i in range(0, len(self.antecedent_indexes)):
                sub_same_indexes = np.intersect1d(self.antecedent_indexes[i], self.consequent_indexes[i], assume_unique=True)
                same_indexes.append(sub_same_indexes)
                sub_num_whole_rule += sub_same_indexes.size
        else:
            same_indexes = np.intersect1d(self.antecedent_indexes, self.consequent_indexes, assume_unique=True)
            sub_num_whole_rule = same_indexes.size
        
        self.num_whole_rule = sub_num_whole_rule
        self.whole_rule_indexes = same_indexes
        #print("Num whole rule ", self.num_whole_rule)
        self.support = self.num_whole_rule/self.antecedent_applicable


    def calc_overall_support(self, df):
        if self.sequence:
            self.calc_overall_support_sequence(df)
        else:
            self.calc_overall_support_non_sequence(df)

    def calc_confidence(self):
        if self.num_antecedent != 0:
            self.confidence = self.num_whole_rule/self.num_antecedent
        else:
            self.confidence = 0.0

    def calc_lift(self):
        self.lift = self.confidence/self.consequent_support

    def get_average_penalty(self, kind):
        penalty = 0
        divisor = 0
        for param in self.rule_dict:
            if kind == "range":
                penalty += self.rule_dict[param].return_bound_amplitude_percent()
            elif kind == "sequence": 
                penalty += self.rule_dict[param].return_sequence_amplitude_percent()
            divisor += 1
        return penalty/divisor


    def run_sequence_penalty(self):
        if self.sequence_penalty:
                s_penalty = self.get_average_penalty("sequence")
                #Need to think about this better. 
                if s_penalty > 0:
                    if self.sequence_penalty_index == 0:
                        self.fitness = self.fitness-(1*(0.1*s_penalty))
                    if self.sequence_penalty_index == 1:
                        self.fitness = self.fitness-(1*(0.5*s_penalty))
                    if self.sequence_penalty_index == 2:
                        earliest, latest, earliest_param_name = self.get_rule_sequence_bounds_and_earliest_param()
                        #Gives a penalty if the function looking too far back at the end. 
                        self.fitness = self.fitness -(1*(0.5*s_penalty) + (0.8*(earliest/self.sequence_limit)))
                    if self.sequence_penalty_index == 3:
                        earliest, latest, earliest_param_name = self.get_rule_sequence_bounds_and_earliest_param()
                        #Gives a penalty if the function looking too far back at the beginning. 
                        self.fitness = self.fitness -(1*(0.5*s_penalty) + (0.8*(latest/self.sequence_limit)))
                    if self.sequence_penalty_index == 4:
                        earliest, latest, earliest_param_name = self.get_rule_sequence_bounds_and_earliest_param()
                        #Gives a penalty if the function looking too far back at the beginning. 
                        self.fitness = self.fitness -(1*(0.1*s_penalty) + (0.8*(latest/self.sequence_limit)))


    def run_range_penalty(self):
        if self.range_penalty:
                r_penalty = self.get_average_penalty("range")
                if r_penalty > 0:
                    if self.range_penalty_index == 0:
                        self.fitness = self.fitness-1*(0.1*r_penalty)
                    if self.range_penalty_index == 1:
                        self.fitness = self.fitness-0.2*(0.1*r_penalty)


    def run_fitness_function(self):
        index = self.fitness_function_index
        if index == 0:
            self.fitness = (2*self.support * (self.num_whole_rule/self.num_consequent))*self.confidence
        if index == 1: 
            self.fitness = (5*self.support * (self.num_whole_rule/self.num_consequent))+self.confidence
        if index == 2: 
            self.fitness = (5*self.support+0.5*self.confidence)
        if index == 3:
            self.fitness = (2*self.support * (self.num_whole_rule/self.num_consequent))*self.confidence*self.lift
        if index == 4:
            self.fitness = (5*self.support * (self.num_whole_rule/self.num_consequent))+0.5*self.confidence
        if index == 5: 
            self.fitness = (5*self.support+0.5*self.confidence+0.1*self.lift)
        if index == 6:
            self.fitness = (2*self.support*2*self.confidence*5*(1-self.lift))
        if index == 7:
            self.fitness = (8*self.support + 5*self.confidence*1*(1-self.lift))
        if index == 8:
            self.fitness = (self.support + self.confidence*1*(1-self.lift))
        if index == 9:
            self.fitness = (2*self.support + self.confidence*1*(1-self.lift))
        if index == 10:
            self.fitness = (2*self.support+(self.num_whole_rule/self.num_consequent)+ 3*self.confidence*1*(1-self.lift))
        if index == 11:
            self.fitness = (2*self.support+(self.num_whole_rule/self.num_consequent)+ 8*self.confidence*1*(1-self.lift))
        if index == 12:
            self.fitness = (1*self.support+5*(self.num_whole_rule/self.num_consequent)+ 5*self.confidence + 0.1*self.lift)

        self.run_sequence_penalty()
        self.run_range_penalty()

    def calc_fitness(self, df):
        self.build_rule_antecedent_query()
        self.build_consequent_query()
        self.calc_antecedent_support(df)
        self.calc_overall_support(df)
        self.calc_confidence()
        self.calc_lift()
        self.run_fitness_function()
   

    #Gets the earliest sequence value (higher number), latest sequence value (lower number), and param with earliest sequence number 
    def get_rule_sequence_bounds_and_earliest_param(self):
        if self.sequence:
            latest = None
            earliest = None 
            earliest_param_name = None
            for item in list(self.rule_dict.keys()):
                sub_latest, sub_earliest = self.rule_dict[item].return_seq_bounds()
                if latest == None:
                    latest = sub_latest
                    earliest = sub_earliest
                    earliest_param_name = item
                else:
                    if sub_earliest > earliest:
                        earliest = sub_earliest
                        earliest_param_name = item
                    if sub_latest < latest:
                        latest = sub_latest
            return earliest, latest, earliest_param_name
        else:
            return False, False, False

    #Gets the earliest sequence value (higher number), latest sequence value (lower number), and param with earliest sequence number 
    def get_outlier_sequence_bounds(self, which, min_max):
        if self.sequence:
            curr_bound = None 
            for item in list(self.rule_dict.keys()):
                sub_latest, sub_earliest = self.rule_dict[item].return_seq_bounds()
                if which == "upper":
                    bound_of_interest = sub_earliest
                else:
                    bound_of_interest = sub_latest
                if curr_bound == None:
                    curr_bound = bound_of_interest
                else:
                    if min_max == "min":
                        if bound_of_interest < curr_bound:
                            curr_bound = bound_of_interest
                    elif min_max == "max":
                        if bound_of_interest > curr_bound:
                            curr_bound = bound_of_interest
            return curr_bound
        else:
            return False 
        
    
    def get_fitness(self):
        return self.fitness
    
    def get_rule_dict(self):
        return self.rule_dict.copy()

    def get_rule_dict_all_numeric(self):
        new_rule_dict = {}
        new_rule_dict["parameters"] = {}
        for param in list(self.rule_dict.keys()):
            #Solve this? 
            new_rule_dict["parameters"][param] = {}
            new_rule_dict["parameters"][param]["lower_bound"] = self.rule_dict[param].curr_lower_bound
            new_rule_dict["parameters"][param]["upper_bound"] = self.rule_dict[param].curr_upper_bound
            if self.sequence:
                new_rule_dict["parameters"][param]["seq_lower_bound"] = self.rule_dict[param].curr_sequence_lower
                new_rule_dict["parameters"][param]["seq_upper_bound"] = self.rule_dict[param].curr_sequence_upper
        new_rule_dict["support"] = self.support
        new_rule_dict["confidence"] = self.confidence
        new_rule_dict["lift"] = self.lift
        new_rule_dict["fitness"] = self.fitness
        return new_rule_dict

    def get_bounds_list(self):
        bounds_list = []
        rule_keys = list(self.rule_dict.copy())
        rule_keys = sorted(rule_keys)
        for key in rule_keys:
            bounds_list.append(self.rule_dict[key].curr_lower_bound)
            bounds_list.append(self.rule_dict[key].curr_upper_bound)
        return bounds_list
    def get_active_parameters(self):
        return self.active_parameters

    def add_parameter(self):
        #Get the parameters we aren't currently using
        non_included_params = list(set(self.parameter_list) - set(self.active_parameters))
        #Pick a random one
        try:
            new_param = random.choice(non_included_params)
            #Init and add it 
            self.rule_dict[new_param] = ga_parameter.parameter(new_param, self.features_dict)
            self.active_parameters.append(new_param)
        except Exception as e:
            pass 
            #This probably means we already have all parameters - so skip 
            #Mutation for now. 


    def subtract_parameter(self):
        #Pick a param in the rule 
        delete_param = random.choice(self.active_parameters)
        self.active_parameters.remove(delete_param)
        self.rule_dict.pop(delete_param)


    def perform_mutation(self, df, kind=None):
        add_chance = self.add_subtract_percent/2
        subtract_chance = add_chance
        change_chance = self.change_percent 
        if kind == None:
            kind_of_mutation = random.choices(["add", "subtract", "change"], weights=[add_chance, subtract_chance, change_chance], k=1)[0]
        else:
            kind_of_mutation = kind
        #START HERE! 
        #Add or subtract 
        if kind_of_mutation == "add":
            self.add_parameter()
            self.last_mutation_type = "add"
        elif kind_of_mutation == "subtract":
            #Only subtract if there is more than one parameter. Otherwise add
            if len(self.active_parameters) < 2:
                self.last_mutation_type = "add"
                self.add_parameter()
            #Otherwise, random choice of add or subtract 
            else:
                self.last_mutation_type = "subtract"
                self.subtract_parameter()
        #Or, change the boundaries 
        else:
            self.last_mutation_type = "change"
            mutate_param = random.choice(self.active_parameters)
            self.rule_dict[mutate_param].mutate()
        return kind_of_mutation
        
    #Get rid of print statements here eventually! 
    def mutate(self, df, kind=None): 
        #Use the percentages to figure out what kinds of mutation to do 
        old_rule_dict = self.rule_dict.copy()
        old_active_params = self.active_parameters.copy()
        kind_of_mutation = self.perform_mutation(df)
        self.calc_fitness(df)
        tries = 0 
        #If we mutated into something not present, try again. 
        if self.antecedent_support <= 0.0 and tries < self.max_mutation_tries:
            tries += 1
            self.rule_dict = old_rule_dict
            self.active_parameters = old_active_params
            self.perform_mutation(df, kind=kind_of_mutation)
            self.calc_fitness(df)

                    
    def print_full(self):
        print(f"Mutation Rate {self.mutation_rate}")
        print(f"Maximum allowed initial parameters {self.init_max_params}")
        print(f"Last Mutation {self.last_mutation_type}")
        print(f"Active parameters {self.active_parameters}")
        for rule in self.active_parameters:
            self.rule_dict[rule].print_name()
            self.rule_dict[rule].print_current()
        self.print_fitness_metrics()

    def print_self(self):
        print(f"Last Mutation {self.last_mutation_type}")
        print(f"Active parameters {self.active_parameters}")
        for rule in self.active_parameters:
            self.rule_dict[rule].print_name()
            self.rule_dict[rule].print_current()

    def print_fitness_metrics(self):
        print(f"Antecedent Support {self.antecedent_support}")
        print(f"Consequent Support {self.consequent_support}")
        print(f"Overall Support {self.support}")
        print(f"Confidence {self.confidence}")
        print(f"Lift {self.lift}")
        print(f"Overall Fitness {self.fitness}")

    def elegant_print(self):
        if self.sequence:
            for item in list(self.rule_dict.keys()):
                print(f"{item}: [{round(self.rule_dict[item].curr_lower_bound, 3)}, {round(self.rule_dict[item].curr_upper_bound, 3)}]  [{self.rule_dict[item].curr_sequence_lower}, {self.rule_dict[item].curr_sequence_upper}]")
        else:
            for item in list(self.rule_dict.keys()):
                print(f"{item}: [{round(self.rule_dict[item].curr_lower_bound, 3)}, {round(self.rule_dict[item].curr_upper_bound, 3)}]")
     

    #I think we need this in order to be able to sort...
    def __lt__(self, other):
         return self.fitness < other.fitness


