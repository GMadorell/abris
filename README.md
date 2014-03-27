#Abris

##Configuration file description
The configuration file is just a simple json that can be easily modified in order to customize the data preprocessing engine.  

Example of a full configuration json:

    {
        "data_model": {
            "age": ["Numeric"],
            "country": ["Categorical"],
            "points_per_match": ["Numeric"],
            "is_awesome": ["Boolean"],
            "paid_price": ["Numeric", "Target"]
        },
        "nan_treatment": {
            "enabled": "True",
            "method": "mean"
        },
        "scaling": {
            "enabled": "True"
        },
        "split": {
            "enabled": "True",
            "train_percentage": 0.7
        }
    }


Below is a brief description of the accepted parameters.  

###Data Model  
This might probably be the most important part of the configuration file.  
Here you describe how the input data will be shaped so it can be processed.

Each feature is described this way:  
    
    "name_of_the_feature": array_of_settings

The array settings can have different values:  

 * Categorical: The feature will be processed and a new dummy feature will be built for each different value of the the original feature.
 * Numeric: Processed as a number. This features can be later scaled.
 * Boolean: Will be binarized into ones and zeros depending on the boolean value.
 * Ignore: This features will be simply dropped out at the cleaning stage.
 * Target: For supervised problems only. This feature will be returned as the last column of the transformed array and will be processed slightly different than the other features. For example, if the feature is categorical, no dummy features will be built but each different value will be assigned a number instead.

Example of a data model:

    "data_model": {
        "age": ["Numeric"],
        "country": ["Categorical"],
        "points_per_match": ["Numeric"],
        "is_awesome": ["Boolean"],
        "paid_price": ["Numeric", "Target"]
    }

### Missing values treatment
Abris supports automatically dealing with missing values of the dataset.  
Supported methods:  

* drop_rows: All the samples containing at least one missing value will be dropped.
* mean: Use the mean of the feature to handle the missing values. This will only work for numeric features, the categorical ones will use the mode.
* median: Fill numeric features with the median value and categorical ones with the mode.
* mode: Fill all features with the mode.

Example of a missing values configuration part:

    "nan_treatment": {
        "enabled": "True",
        "method": "mean"
    }

### Scaling
Abris supports auto scaling numerical features.  
Example:

    "scaling": {
        "enabled": "True"
    }

### Train / Test split.
Abris also supports automatically splitting the dataset in two, in a randomized fashion.  
Example:

    "split": {
        "enabled": "True",
        "train_percentage": 0.7
    }