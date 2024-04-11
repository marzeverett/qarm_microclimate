# qarm_microclimate
Code for quantitative association rule mining with genetic algorithms for microclimate data. 


## Population 
The population expects the following arguments: 
* parameter_dictionary: A dictionary with the following parameters: (TODO -- fill in)
    
    * mutation_rate: the percent chance of mutation 
    * mutation_amount: the upper limit of mutation amount(MORE)
    * range_restriction: False, or a % of the standard deviation of the value that you want to restrict the range to. 
    * range_penalty: True if you want to penalize the range length. If True, make sure to set the range penalty index
    * intial_rule_limit: Upper limit on how many antecendents you'll allow in an initialized rule
    * index_key: key for individual records
    * consequent_key: key for consequent feature
    * add_subtract_percent: percent of chance for a mutation to be add/subtract. This plus change_percent should = 100. 
    * change percent: percent of chance for a mutatino to be a change. This plus add_subtract_percent should = 100
    * max_mutation_tries: max amount of times to try mutating a rule before giving up if the mutation doesn't satisfy criteria.
    * population_size: size of the population
    * top_rules: number of top rules to keep across generations
    * generations: number of generations to run
    * dominance: If True, kill rules dominated by other rules
    * sequence: True if the data is sequence data and we should be evolving sequences. 
    * sequence_limit: False of size limit of sequence as an int, representing the max span of timesteps
    * sequence_offset: Number of time steps back from consequent the sequence should start from. 
    * sequence_penalty: If true, will impose a penalty for long sequences. If True, sequence penalty index should be set
    * diversify_top_rules: If True, top rules kept should be dissimilar from one another 
    * reseed_from_best: If True, population reseeds killed rules from the best performers. If False, it randomly reseeds them. This can impact diversity. 
    * fitness_function_index: Integer index of fitness function to be used
    * sequence_penalty_index: index of sequence penalty to used, if applicable
    * range_penalty_index: index of range penalty to used, if applicable 

* A consquent dictionary of the form:

        consequent_dict = {
                "name": "[feature name that represents the consequent]",
                "type": "[type of data. Currently supports boolean]",
                "upper_bound": [Consequent upper bound. Currently supports the value of 1],
                "lower_bound": [Consequent lower bound. Currently supports the value of 1]
            }

* feature_dict - A dictionary with entries of the form: 

        "name of feature": {
                "name": "name_of_feature",
                "type": "continuous" #Currently only supported type right now
            },

    for all the features you want to use as antecedents. 

* train_df - the pandas dataframe with the training data. 


## Running an Experiment 

To run an experiment:

1. First, read in your dataset
2. Then, create your parameter dictionary 
3. Create your antecedent and consequent dictionaries 
4. Split your dataset into test and training 
5. Identify your consequent kkey 
6. Create a population
7. Use the run_experiment property of the population
8. Create a filename for your top rules
9. Use the ga_predictor class with the eval_top_rules property to eval it. 