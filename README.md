# qarm_microclimate
Code for quantitative association rule mining with genetic algorithms for microclimate data. 


## Population 
The population expects the following arguments: 
* parameter_dictionary: A dictionary with the following parameters: (TODO -- fill in)

* A consquent dictionary of the form:

        consequent_dict = {
                "name": "[feature name that represents the consequent]",
                "type": "[type of data. Currently supports boolean]",
                "upper_bound": [Consequent upper bound. Currently supports the value of 1],
                "lower_bound": [Consequent lower bound. Currently supports the value of 1]
            }

* list_features_dict - A list of dictionaries of the form: 

        "name of feature": {
                "name": "name_of_feature",
                "type": "continuous" #Currently only supported type right now
            },

    for all the features you want to use as antecedents. 

* key - string name of feature that represents the consquent. Redundant and needs to change. 


* train_df - the pandas dataframe with the training data. 