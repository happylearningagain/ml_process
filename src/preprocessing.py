import util as utils
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

config = utils.load_config()

def main():
    
    #load train, test set
    train_data, test_data = loadData(config["traintest_set_path"])
    #oversampling train data
    x_train, y_train = RandomOverSampler(random_state = 112).fit_resample(train_data.drop(config["label"], axis = 1), train_data[config["label"]])
    
    #impute missing value in train data (opt=1 -> train data, opt=2 -> test data)
    isnull_column = ['Bare_nuclei']
    x_train = imputeMisVal(x_train, isnull_column, 1)
    
    #normalize/standardize train data
    x_train_clean = standardizerData(x_train, 1)
 
    #impute missing value in test data (opt=1 -> train data, opt=2 -> test data)
    x_test = imputeMisVal(test_data[config["predictors"]], isnull_column, 2)
    
    #normalize/standardize train data
    x_test_clean = standardizerData(x_test, 2)
   
    #dump clean train set
    utils.pickle_dump(x_train_clean, config["traintest_clean_set_path"][0])
    utils.pickle_dump(y_train, config["traintest_clean_set_path"][1])
    utils.pickle_dump(x_test_clean, config["traintest_clean_set_path"][2])


#load train test data
def loadData(config_data: dict) -> pd.DataFrame:
    x_train = utils.pickle_load(config_data[0])
    y_train = utils.pickle_load(config_data[1])
    
    x_test = utils.pickle_load(config_data[2])
    y_test = utils.pickle_load(config_data[3])
    
    train_data = pd.concat([x_train, y_train], axis = 1)
    test_data = pd.concat([x_test, y_test], axis = 1)
    
    return train_data, test_data

#handling missing value
def imputeMisVal(X, numerical_column, opt):
    #get columns containing missing value
    missingval_data = pd.DataFrame(X[numerical_column])
    if opt == 1 :
        #use median of data to impute missing value in train data
        imputer = SimpleImputer(missing_values = np.nan,strategy = "median")
        
        #fit imputer
        imputer.fit(missingval_data)
        utils.pickle_dump(imputer, config["imputer"])
    else:
        #get train data imputer to fill missing value in test data 
        imputer = utils.pickle_load(config["imputer"])

    #transform
    imputed_data = imputer.transform(missingval_data)
    numerical_data_imputed = pd.DataFrame(imputed_data.astype(int))

    numerical_data_imputed.columns = numerical_column
    numerical_data_imputed.index = missingval_data.index
    X[numerical_column] = numerical_data_imputed

    return X                       

def standardizerData(data, opt):
    """
    Fungsi untuk melakukan standarisasi data
    :param data: <pandas dataframe> sampel data
    :param opt: <int> 1 untuk train data, 2 untuk test data
    :return standardized_data: <pandas dataframe> sampel data standard
    :return standardizer: method untuk standardisasi data
    """
    data_columns = data.columns  # agar nama kolom tidak hilang
    data_index = data.index  # agar index tidak hilang

    if opt == 1 :
        # buat (fit) standardizer
        standardizer = StandardScaler()
        standardizer.fit(data)
        utils.pickle_dump(standardizer, config["standardizer"])
    else:
        standardizer = utils.pickle_load(config["standardizer"])

    # transform data
    standardized_data_raw = standardizer.transform(data)
    standardized_data = pd.DataFrame(standardized_data_raw)
    standardized_data.columns = data_columns
    standardized_data.index = data_index

    return standardized_data


if __name__ == "__main__":
    main()