import pandas as pd
import numpy as np
import util as utils
import preprocessing as prep
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

def main():
    config = utils.load_config()
    
    #load clean train-test data
    train_clean, test_clean = prep.loadData(config["traintest_clean_set_path"])
    x_train = train_clean[config["predictors"]]
    y_train = train_clean[config["label"]]
    
    #initialized models
    knn = KNeighborsClassifier()
    logreg = LogisticRegression(random_state = 123) 
    svm = SVC()
    rf = RandomForestClassifier()

    #set parameters for each models
    models = {"classifier" : [knn, 
                             logreg,
                             svm, 
                             rf],
              "params": [{"n_neighbors": [3,5,7,9], "p":[1,2]},
                         {"solver" :["lbfgs", "liblinear"],"C":np.logspace(-3, 3,50)},
                         {"kernel": ["linear","poly", "rbf", "sigmoid"], "C": np.logspace(-5,5,50)},
                         {"n_estimators": [10,20,30,40,50,60,70,80,90,100],"max_features": ['sqrt', 'log2']}]
             }

    #search best parameter
    cv_models = []
    best_param = []
    acc_cv = []
    j = 0
    for i in models["classifier"]:
        cv_models.append(searchBestParam(x_train, y_train.values.ravel(), models["classifier"][j], models["params"][j]))
        best_param.append(cv_models[j].best_params_)
        acc_cv.append(cv_models[j].best_score_) 
        j+=1

    # fit models in training data using best parameter
    knn = KNeighborsClassifier(n_neighbors = best_param[0]['n_neighbors'], 
                               p = best_param[0]['p'])
    logreg = LogisticRegression(solver = best_param[1]['solver'],
                                C = best_param[1]['C'])
    svm = SVC(kernel = best_param[2]['kernel'], 
              C=best_param[2]['C'])    
    rf = RandomForestClassifier(max_features = best_param[3]['max_features'],
                                n_estimators = best_param[3]['n_estimators'],
                                random_state = 123)                            
    knn.fit(x_train,y_train.values.ravel())
    logreg.fit(x_train,y_train.values.ravel())
    svm.fit(x_train,y_train.values.ravel())
    rf.fit(x_train,y_train.values.ravel())

    #predict train data using fitted models
    predicted_knn = knn.predict(x_train)
    predicted_logreg = logreg.predict(x_train)
    predicted_svm = svm.predict(x_train)
    predicted_rf = rf.predict(x_train)

    classifier = [knn, logreg, svm, rf]
    acc_train = [knn.score(x_train,y_train), logreg.score(x_train,y_train), svm.score(x_train,y_train), rf.score(x_train,y_train)]
    indexes = ["KNN", "Logistic Regression", "SVM", "Random Forest"]

    summary_df = pd.DataFrame({ "Best Parameter": best_param,
                            "Accuracy CV": acc_cv,
                            "Accuracy Train": acc_train
                           },
                          index = indexes)  
    print(summary_df)

    #get best model
    index_best = acc_cv.index(np.max(acc_cv))
    best_classifier_model = classifier[index_best]
    utils.pickle_dump(best_classifier_model, config["production_model_path"])

#search best model parameters using gridSearch cross validation
def searchBestParam (x_train, y_train, model_estimator, params):
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
                           cv = 5,
                           verbose = 5,
                           scoring = "accuracy")
    model_cv.fit(x_train, y_train)
    
    return model_cv



if __name__ == "__main__":
    main()