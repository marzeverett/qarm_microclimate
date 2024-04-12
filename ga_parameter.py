import json
import pandas as pd 
import random 
import math 
import copy 
import time 


#Parameter has a lower and upper bound - Changes based on mutation
#Also has an allowed range
############################################ PARAMETER CLASS ##################################################
class parameter:
    def __init__(self, name, features_dict):
        self.name = name
        #Give the name of the feature
        feature_dict = features_dict[name]
        #Get type 
        self.type = feature_dict["type"]
        #Get the upper and lower allowable bound for feature value
        self.upper_bound = feature_dict["upper_bound"]
        self.lower_bound = feature_dict["lower_bound"]
        self.range_restriction = feature_dict["range_restriction"]
        self.max_mutation_tries = feature_dict["max_mutation_tries"]
        self.mutation_amount = feature_dict["mutation_amount"]
        #For sequence extension 
        if feature_dict["sequence"] == True:
            self.sequence = True
            self.sequence_limit = feature_dict["sequence_limit"]
            self.sequence_offset = feature_dict.get("sequence_offset", False)
        else:
            self.sequence = False 
        
        if self.type == "continuous":
            self.mean = feature_dict["mean"]
            self.stdev = feature_dict["stdev"]
        self.random_init()

    def random_init(self):
        if self.type == "continuous" or self.type == "nominal":
            self.random_bounds()
        else:
            self.random_bool()
        if self.sequence:
            self.random_sequence() 

    
    def random_sequence(self):
        start_int = 0
        if self.sequence_offset:
            start_int = self.sequence_offset
        bound_1 = random.randint(start_int, start_int+self.sequence_limit)
        bound_2 = random.randint(start_int, start_int+self.sequence_limit)
        if bound_1 > bound_2:
            self.curr_sequence_upper = bound_1
            self.curr_sequence_lower = bound_2
        else:
            self.curr_sequence_upper = bound_2
            self.curr_sequence_lower = bound_1

        #CHANGE HERE
        if self.sequence_limit == 0:
            self.curr_sequence_upper = 0
            self.curr_sequence_lower = 0

    def random_bool(self):
        self.upper_bound = random.randint((0,1))
        self.lower_bound = self.upper_bound
    
    def random_bounds(self):
        start_val = self.get_random_value_within_bounds()
        mutation = self.get_bound_change(start_val)
        end_val = start_val + mutation
        if start_val < end_val:
            self.curr_upper_bound = end_val
            self.curr_lower_bound = start_val
        else:
            self.curr_upper_bound = start_val
            self.curr_lower_bound = end_val
        
        
        
    #Return a positive or negative change from a bound start value
    #While does NOT violate the rules. 
    def get_bound_change(self, start_value):
        #Get the mutation amount 
        mutation_amount = self.get_random_mutation_amount()
        if (start_value + mutation_amount) > self.upper_bound:
            #If its the max, change direction
            if start_value == self.upper_bound:
                mutation_amount = mutation_amount*-1
            #Otherwise, set it to the max
            else:
                bound_change = self.upper_bound - start_value
        if (start_value + mutation_amount) < self.lower_bound:
            bound_change = self.lower_bound - start_value

        else:
            bound_change = mutation_amount
        return bound_change
        

    def get_random_mutation_amount(self):
        #Random positive or negative amount
        percent_change = self.mutation_amount/100
        sign = random.choice([-1, 1])
        mutation_val = random.uniform(0, percent_change*self.stdev)
        if self.type == "nominal":
            mutation_val = math.ceil(mutation_val)
        mutation_val = mutation_val * sign 
        return mutation_val

    def get_random_value_within_bounds(self):
        value = random.uniform(self.lower_bound, self.upper_bound)
        if self.type == "nominal":
            value = math.ceil(value)
        return value


    def mutate_value(self):
        if self.type == "continuous" or self.type == "nominal":
            tries = 0
            success = 0
            while success == 0 and tries < self.max_mutation_tries:
                success = self.mutate_bounds()
                tries += 1 
        else:
            self.mutate_bool()

    def mutate(self):
        #If we are also mutating sequences
        if self.sequence:
            choice = random.choice(["sequence", "value"])
            if choice == "sequence":
                #CHANGE HERE
                if self.sequence_limit != 0:
                    self.mutate_sequence()
                else:
                    self.mutate_value()
            else:
                self.mutate_value()
        #Otherwise just mutate the rule value 
        else:
            self.mutate_value()
        

    def mutate_bounds(self):
        old_lower = self.curr_lower_bound
        old_upper = self.curr_upper_bound
        choice = random.choice(["lower", "upper"])
        if choice == "upper":
            change = self.get_bound_change(self.curr_upper_bound)
            self.curr_upper_bound = self.curr_upper_bound + change 
        else:
            change = self.get_bound_change(self.curr_lower_bound)
            self.curr_lower_bound = self.curr_lower_bound + change
        if self.curr_lower_bound > self.curr_upper_bound:
            temp = self.curr_upper_bound
            self.curr_upper_bound = self.curr_lower_bound
            self.curr_lower_bound = temp 

        #If this violates the rules, don't mutate it! 
        #CHange here 
        #Check range restriction, if necessary 
        if self.range_restriction:
            if self.curr_upper_bound - self.curr_lower_bound > (self.range_restriction/100)*self.stdev:
                self.curr_lower_bound = old_lower
                self.curr_upper_bound = old_upper
                return False
        return True
        
    def mutate_bool(self):
        if self.curr_lower_bound == 0:
            self.curr_lower_bound = 1
        else:
            self.curr_lower_bound = 0
        self.curr_upper_bound = self.curr_lower_bound

    
    def mutate_sequence(self):
        #Mutate 1-2 days at a time, only. 
        tries = 0
        success = 0
        old_curr_seq_upper = self.curr_sequence_upper
        old_curr_seq_lower = self.curr_sequence_lower
        while success == 0 and tries < self.max_mutation_tries:
            tries += 1
            #Eventually may want to change this. 
            days_to_mutate = random.choice([1,2,-1,-2])
            bound_to_mutate = random.choice(["upper", "lower"])
            if bound_to_mutate == "upper":
                upper_bound = self.curr_sequence_upper + days_to_mutate
                lower_bound = self.curr_sequence_lower
            else:
                lower_bound = self.curr_sequence_lower + days_to_mutate
                upper_bound = self.curr_sequence_upper
            if lower_bound > upper_bound:
                temp = lower_bound
                lower_bound = upper_bound
                upper_bound = temp
            if lower_bound < 0:
                lower_bound = 0
            if self.sequence_limit:
                #If our sequence is too long 
                if (upper_bound - lower_bound) > self.sequence_limit:
                    success = 0
                    #Change is here!
                    self.curr_sequence_upper = old_curr_seq_upper
                    self.curr_sequence_lower = old_curr_seq_lower
            if self.sequence_offset:
                #If we went below our offset - bad. 
                if lower_bound < self.sequence_offset or lower_bound < 0:
                    success = 0
                    #Change is here!
                    self.curr_sequence_upper = old_curr_seq_upper
                    self.curr_sequence_lower = old_curr_seq_lower
            else:
                success = 1
                self.curr_sequence_lower = lower_bound
                self.curr_sequence_upper = upper_bound

            

        # if success == 0:
        #     print("Could not successfully mutate")

    #Return bounds 
    def return_bounds(self):
        #Returns lower, upper bound in that order 
        return self.curr_lower_bound, self.curr_upper_bound

    def return_seq_bounds(self):
        return self.curr_sequence_lower, self.curr_sequence_upper

    def return_bounds_difference(self):
        return self.curr_upper_bound - self.curr_lower_bound

    #Return bound ranges 
    def return_bound_range(self):
        return self.curr_upper_bound - self.curr_lower_bound

    def return_sequence_range(self):
        if self.sequence: 
            return self.curr_sequence_upper - self.curr_sequence_lower
        else:
            return False 

    def return_bound_amplitude_percent(self):
        if not self.range_restriction:
            total_range = self.upper_bound - self.lower_bound
        else:
            total_range = 2*self.stdev*self.range_restriction
        current_range = self.curr_upper_bound - self.lower_bound
        if total_range > 0:
            #Percent of total range covered. 
            return current_range/total_range
        else:
            return 0

    def return_sequence_amplitude_percent(self):
        if self.sequence_limit:
            total_range = self.sequence_limit
        else:
            #Dummy -- not sure what to do here 
            total_range = 1
        curr_range = self.curr_sequence_upper - self.curr_sequence_lower
        if total_range > 0:
            return curr_range/total_range
        else:
            return 0 

    def return_name(self):
        return self.name 

    def print_full(self):
        print(f"Name: {self.name}")
        print(f"Type: {self.type}")
        print(f"Min Lower Bound: {self.lower_bound}")
        print(f"Max Upper Bound: {self.upper_bound}")
        print(f"Curr Lower Bound {self.curr_lower_bound}")
        print(f"Curr Upper Bound {self.curr_upper_bound}")
        if self.sequence:
            print(f"Curr Sequence Upper Bound {self.curr_sequence_upper}")
            print(f"Curr Sequence Lower Bound {self.curr_sequence_lower}")
        print(f"Mutation Amount: {self.mutation_amount}")
        print(f"Range Restriction {self.range_restriction}")
        if self.type == "continuous" or self.type == "nominal":
            print(f"Mean {self.mean}")
            print(f"Standard Deviation {self.stdev}")
        print()


    def print_current(self):
        print(f"Curr Lower Bound {self.curr_lower_bound}")
        print(f"Curr Upper Bound {self.curr_upper_bound}")
        if self.sequence:
            print(f"Curr Sequence Upper Bound {self.curr_sequence_upper}")
            print(f"Curr Sequence Lower Bound {self.curr_sequence_lower}")
        print()

    def print_name(self):
        print(f"Name {self.name}")
        


