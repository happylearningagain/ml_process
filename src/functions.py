import pandas as pd
import numpy as np
import util as utils
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


#load dataset
def open_dataset(config: dict):
    dataset = pd.read_csv(config['dataset_path'], names = config['column_name'], header= None)
    #drop duplicates
    dataset = dataset[config['used_columns']].drop_duplicates()

    return dataset
    
#data defense
def data_checking(input_data, config):
    # check input length
    input_length = len(input_data)
    
    #check input data type
    assert input_data.select_dtypes("int").columns.to_list() == config["int_columns"], "an error occurs in int column(s)."
    assert input_data.select_dtypes("float").columns.to_list() == config["float_columns"], "an error occurs in float column(s)."
    
    #check data range
    assert input_data[config["int_columns"][0]].between(config["Clump_thickness"][0], config["Clump_thickness"][1]).sum() == input_length, "an error occurs in Clump thickness range."
    assert input_data[config["int_columns"][1]].between(config["Uniformity_of_cell_size"][0], config["Uniformity_of_cell_size"][1]).sum() == input_length, "an error occurs in Uniformity of cell size range."
    assert input_data[config["int_columns"][2]].between(config["Uniformity_of_cell_shape"][0], config["Uniformity_of_cell_shape"][1]).sum() == input_length, "an error occurs in Uniformity of cell shape range."
    assert input_data[config["int_columns"][3]].between(config["Marginal_adhesion"][0], config["Marginal_adhesion"][1]).sum() == input_length, "an error occurs in Marginal adhesion range."
    assert input_data[config["int_columns"][4]].between(config["Single_epithelial_cell_size"][0], config["Single_epithelial_cell_size"][1]).sum() == input_length, "an error occurs in Single epithelial cell size range."
    assert input_data[config["float_columns"][0]].between(config["Bare_nuclei"][0], config["Bare_nuclei"][1]).sum() == input_length, "an error occurs in Bare nuclei range."
    assert input_data[config["int_columns"][6]].between(config["Bland_chromatin"][0], config["Bland_chromatin"][1]).sum() == input_length, "an error occurs in Bland chromatin range."
    assert input_data[config["int_columns"][7]].between(config["Normal_nucleoli"][0], config["Normal_nucleoli"][1]).sum() == input_length, "an error occurs in Normal nucleoli range."
    assert input_data[config["int_columns"][8]].between(config["Mitoses"][0], config["Mitoses"][1]).sum() == input_length, "an error occurs in Mitoses range."

#load train test data
def load_data(config_data: dict) -> pd.DataFrame:
    x_train = utils.pickle_load(config_data[0])
    y_train = utils.pickle_load(config_data[1])
    
    x_test = utils.pickle_load(config_data[2])
    y_test = utils.pickle_load(config_data[3])
    
    train_data = pd.concat([x_train, y_train], axis = 1)
    test_data = pd.concat([x_test, y_test], axis = 1)
    
    return train_data, test_data

#normalize / scaling data 
def standardizerData(data):
    """
    Fungsi untuk melakukan standarisasi data
    :param data: <pandas dataframe> sampel data
    :return standardized_data: <pandas dataframe> sampel data standard
    :return standardizer: method untuk standardisasi data
    """
    data_columns = data.columns  # agar nama kolom tidak hilang
    data_index = data.index  # agar index tidak hilang

    # buat (fit) standardizer
    standardizer = StandardScaler()
    standardizer.fit(data)

    # transform data
    standardized_data_raw = standardizer.transform(data)
    standardized_data = pd.DataFrame(standardized_data_raw)
    standardized_data.columns = data_columns
    standardized_data.index = data_index

    return standardized_data, standardizer

#search best model parameters using gridSearch cross validation
def searchBestParam (x_train, y_train, model_estimator, params, fold):
    """
    fungsi untuk train data menggunakan grid search cross validation
    :param x_train: <pandas DataFrame> input data yang akan di training
    :param y_train: <pandas Dataframe> target data
    :param model_estimator:<list> list estimator
    :param params: <list> list parameter yang akan dieksperimenkan
    :fold: <int> jumlah fold yang akan digunakan pada cross validation
    :scoring: <str> tipe scoring
    """
    
    model_cv = GridSearchCV(estimator = model_estimator,
                           param_grid = params,
                           cv = fold,
                           n_jobs = -1,
                           verbose = 5,
                           scoring = "accuracy")
    model_cv.fit(x_train, y_train)
    
    return model_cv

#choose best model   
def bestModel(list_model: dict):
    best_parameter = list_model.best_param
    acc_train = list_model.acc_train
    acc_cv = list_model.acc_cv
    indexes = list_model.model_name
    summary_df = pd.DataFrame({ "Best Parameter": best_parameter,
                                "Accuracy CV": acc_cv,
                                "Accuracy Train": acc_train
                                },
                                index = indexes)

    best_model = list_model.acc_cv == max(acc_cv)
    return summary_df, best_model

