import os
import numpy as np
import pandas as pd
from marketing_code import UsedFunctions

if __name__ == '__main__':

    # define parameters
    filepath = os.getcwd()
    train_name = 'marketing_training.csv'
    test_name = 'marketing_test.csv'
    # train_filepath = os.path.join(filepath, train_name)
    # test_filepath = os.path.join(filepath, test_name)

    train_only = True  # toggle to True for final model, False for training and testing
    # drop_columns = None
    drop_columns = ['pmonths']
    train_size = 0.7
    random_state = 1
    n_estimators = 1000
    learning_rate = 0.2

    algorithm_model = 'xgboost'  # options: 'xgboost', 'random forest', 'lgb'

    # read in training and testing datasets
    df_train, df_test = UsedFunctions.read_data(filepath, train_name, test_name)

    # wrangle data to be passed to models. add columns to be dropped if desired
    train_dataset = UsedFunctions.wrangle_data(df_train, drop_columns)
    test_dataset = UsedFunctions.wrangle_data(df_test, drop_columns)

    # train only or train and test models?
    if train_only:
        # apply specified model to predict response on marketing_test
        test_predictions = UsedFunctions.model_train_and_prediction(train_dataset, test_dataset, algorithm_model)
        # see percent predicted as 1
        predict_perc = 100 * (np.count_nonzero(test_predictions == 1)) / len(test_predictions)

        # add test_predictions as column to marketing_test_predicted.csv
        # add predictions to test_dataset
        # test_dataset['responded'] = test_predictions.tolist()
        df_test_predictions = pd.DataFrame({'responded': test_predictions.tolist()})

        #  convert 0/1 to no/yes
        df_test_predictions['responded'] = df_test_predictions['responded'].map({1: 'yes', 0: 'no'})

        # write dataframe to csv
        df_test_predictions.to_csv('{0}/{1}'.format(filepath, 'marketing_test_predictions.csv'))

    else:
        # when I want to see which model works best on marketing_training, use lines below
        train_and_test = True
        model, predict_responded, test_x, test_y = UsedFunctions.model_train_and_prediction(train_dataset, test_dataset,
                                                                              algorithm_model, train_and_test)

        # evaluate models
        cm = UsedFunctions.confusion_matrix_results(test_y, predict_responded)
        print(cm)