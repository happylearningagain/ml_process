import pandas as pd
import util as utils
from sklearn.model_selection import train_test_split

config = utils.load_config()

#load dataset
def openDataset():
    dataset = pd.read_csv(config['dataset_path'], names = config['column_name'], header= None)
    #drop duplicates
    dataset = dataset[config['used_columns']].drop_duplicates()

    return dataset

#split train-test set
def splitTrainTest(data):
    x = pd.DataFrame(data[config["predictors"]])
    y = pd.DataFrame(data[config["label"]])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, 
                                                        random_state = 123, stratify = y)
 
    utils.pickle_dump(x_train[config["predictors"]], config["traintest_set_path"][0])
    utils.pickle_dump(y_train[config["label"]], config["traintest_set_path"][1])

    utils.pickle_dump(x_test[config["predictors"]], config["traintest_set_path"][2])
    utils.pickle_dump(y_test[config["label"]], config["traintest_set_path"][3])

#data defense
def dataChecking(input_data):
    # check input length
    input_length = len(input_data)
    
    #check input data type
    assert input_data.select_dtypes("int").columns.to_list() == config["int_columns"], "an error occurs in int column(s)."
    
    #check data range
    assert input_data[config["int_columns"][0]].between(config["Clump_thickness"][0], config["Clump_thickness"][1]).sum() == input_length, "an error occurs in Clump thickness range."
    assert input_data[config["int_columns"][1]].between(config["Uniformity_of_cell_size"][0], config["Uniformity_of_cell_size"][1]).sum() == input_length, "an error occurs in Uniformity of cell size range."
    assert input_data[config["int_columns"][2]].between(config["Uniformity_of_cell_shape"][0], config["Uniformity_of_cell_shape"][1]).sum() == input_length, "an error occurs in Uniformity of cell shape range."
    assert input_data[config["int_columns"][3]].between(config["Marginal_adhesion"][0], config["Marginal_adhesion"][1]).sum() == input_length, "an error occurs in Marginal adhesion range."
    assert input_data[config["int_columns"][4]].between(config["Single_epithelial_cell_size"][0], config["Single_epithelial_cell_size"][1]).sum() == input_length, "an error occurs in Single epithelial cell size range."
    assert input_data[config["int_columns"][0]].between(config["Bare_nuclei"][0], config["Bare_nuclei"][1]).sum() == input_length, "an error occurs in Bare nuclei range."
    assert input_data[config["int_columns"][6]].between(config["Bland_chromatin"][0], config["Bland_chromatin"][1]).sum() == input_length, "an error occurs in Bland chromatin range."
    assert input_data[config["int_columns"][7]].between(config["Normal_nucleoli"][0], config["Normal_nucleoli"][1]).sum() == input_length, "an error occurs in Normal nucleoli range."
    assert input_data[config["int_columns"][8]].between(config["Mitoses"][0], config["Mitoses"][1]).sum() == input_length, "an error occurs in Mitoses range."


def main():
    data = openDataset()
    splitTrainTest(data)


if __name__ == "__main__" :
    main()