import pandas as pd
import numpy as np
import os

class UsedFunctions():
    def __init__(self, split_features_labels, use_random_forest, use_xgboost, use_lgb, confusion_matrix_results):
        self.split_features_labels = split_features_labels
        self.use_random_forest = use_random_forest
        self.use_xgboost = use_xgboost
        self.use_lgb = use_lgb
        self.confusion_matrix_results = confusion_matrix_results

    def read_data(dir_name: str, name_train_data: str, name_test_data: str):
        """
        :param dir_name: filepath where data is located
        :param name_train_data: name of train dataset
        :param name_test_data: name of test dataset
        :return: dataframe of data from CSV
        """

        train_file = os.path.join(dir_name, name_train_data)
        test_file = os.path.join(dir_name, name_test_data)

        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        # df_test first column is index so it can be dropped
        test_data = test_data.drop(test_data.columns[0], axis=1)

        return train_data, test_data

    def wrangle_data(data, columnsToDrop=None):
        """
        :param data: dataframe
        :param columnsToDrop: Optional list of column names to remove as a feature
        :return: dataframe with data formatted to fit multiple algorithms
        """

        # change responded values to 0 / 1 for no / yes since it's binary for train data
        if 'responded' in data.columns:
            data['responded_binary'] = data['responded'].eq('yes').mul(1)

            # drop responded column
            data = data.drop(columns='responded')

        # drop columns that exist in columnsToDrop
        if columnsToDrop is not None:
            data = data.drop(columns=columnsToDrop)

        # replace missing custAge values (if it exists)
        if columnsToDrop is None:
            # avg age for students and retired people are significantly lower and higher, respectively, than the avg age
            # replace missing age for students and retired people with the average age of students and retirees, respectively
            # replace all other missing values with avg of all
            temp_data = data.groupby('profession', as_index=False)['custAge'].mean()
            avg_age = round(temp_data)
            avg_age_all = round(data['custAge'].mean())
            for i in range(len(data)):
                if pd.isnull(data.loc[i, 'custAge']) == True:
                    if data['profession'][i] == 'student':
                        data['custAge'][i] = avg_age.loc[avg_age['profession'] == 'student', 'custAge'].iloc[0]
                    elif data['profession'][i] == 'retired':
                        data['custAge'][i] = avg_age.loc[avg_age['profession'] == 'retired', 'custAge'].iloc[0]
                    else:
                        data['custAge'][i] = avg_age_all
        elif 'custAge' not in columnsToDrop and 'profession' not in columnsToDrop:
            # avg age for students and retired people are significantly lower and higher, respectively, than the avg age
            # replace missing age for students and retired people with the average age of students and retirees, respectively
            # replace all other missing values with avg of all
            temp_data = data.groupby('profession', as_index=False)['custAge'].mean()
            avg_age = round(temp_data)
            avg_age_all = round(data['custAge'].mean())
            for i in range(len(data)):
                if pd.isnull(data.loc[i, 'custAge']) == True:
                    if data['profession'][i] == 'student':
                        data['custAge'][i] = avg_age.loc[avg_age['profession'] == 'student', 'custAge'].iloc[0]
                    elif data['profession'][i] == 'retired':
                        data['custAge'][i] = avg_age.loc[avg_age['profession'] == 'retired', 'custAge'].iloc[0]
                    else:
                        data['custAge'][i] = avg_age_all
        elif 'custAge' not in columnsToDrop & 'profession' in columnsToDrop:
            # replace missing custAge with avg age
            avg_age_all = round(data['custAge'].mean())
            for i in range(len(data)):
                if pd.isnull(data.loc[i, 'custAge']) == True:
                    data['custAge'][i] = avg_age_all

        # schooling has an option of unknown, so all NaN values will be changed to unknown
        if columnsToDrop is not None:
            if 'profession' not in columnsToDrop:
                for i in range(len(data)):
                    if pd.isnull(data.loc[i, 'schooling']) == True:
                        data['schooling'][i] = 'unknown'

        # pmonths and pdays represent the same thing (in different value formats), so pdays is dropped
        # pmonths is kept because the difference between not being contacted (999) and contacted in months is more drastic
        if columnsToDrop is not None:
            if 'pdays' not in columnsToDrop and 'pmonths' not in columnsToDrop:
                data = data.drop(columns='pdays')
            elif 'pmonths' in columnsToDrop:
                # make pdays larger to create larger differentiation between no contact and days since contact
                data['pdays'] = data['pdays'].replace(999, 99999)
        else:
            data = data.drop(columns='pdays')

        # all columns of type object are converted to categorical then one-hot encoded
        objectColumns = data.dtypes[data.dtypes == np.object]
        for obj in objectColumns.index.values:
            data[obj] = data[obj].astype('category')

        data = pd.concat([data, pd.get_dummies(data)], axis=1)

        # drop category columns and the duplicated columns (from running get_dummies)
        data = data.drop(columns=objectColumns.index.values)
        data = data.loc[:, ~data.columns.duplicated()]

        # testing dataset is missing schooling_illiterate and default_yes. add both to testing_dataset with values of 0.0
        if 'schooling_illiterate' not in data:
            # col_illeterate = 0.0
            data.insert(loc=31, column='schooling_illiterate', value=0.0)
            data.insert(loc=37, column='default_yes', value=0.0)
            # data['default_yes'] = 0.0

        return data

    def split_features_labels(self, data, toTrainAndTest=False, train_split=None, state_random=None):
        """
        :param data: dataframe
        :param toTrainAndTest: Optional, default is False which indicates the data input is only used for training. False indicates this dataset is used for training and testing
        :param train_split: Optional, default is None. train_split is used to select the train size
        :param state_random: Optional, default is None. state_random is used to select a random state
        :return: dataframe of features and labels
        """

        from sklearn.model_selection import train_test_split

        features = data.drop(columns='responded_binary')
        labels = data['responded_binary']

        if toTrainAndTest:
            train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                        train_size=train_split,
                                                                                        random_state=state_random)
            return train_features, train_labels, test_features, test_labels
        else:
            return features, labels

    def use_random_forest(self, train_features, train_labels, n, state_random, test_features=None):
        """
        :param train_features: dataframe of features for training
        :param train_labels:  dataframe of labels for training
        :param n: number of trees
        :param state_random: used to select random state
        :param test_features: Optional, default is None. If not None, dataframe of features is used for predicting labels
        :return: model and label predictions (if test_features is not None)
        """

        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=n, random_state=state_random, class_weight='balanced')
        model.fit(train_features, train_labels)

        if test_features is None:
            return model
        else:
            # predict on test portion of train data
            predictions = model.predict(test_features)
            return model, predictions

    def use_xgboost(self, train_features, train_labels, state_random, rate_learning=0.1, weight=9, test_features=None):
        """
        :param train_features: dataframe of features for training
        :param train_labels:  dataframe of labels for training
        :param state_random: used to select random state
        :param rate_learning: Optional, default is 0.1. Used to select the learning rate for the model
        :param weight: Optional, default is 9. Used to select the imbalance ratio of labels
        :param test_features: Optional, default is None. If not None, dataframe of features is used for predicting labels
        :return: model and label predictions (if test_features is not None)
        """

        import xgboost as xgb
        model = xgb.XGBClassifier(random_state=state_random, learning_rate=rate_learning, scale_pos_weight=weight)
        model.fit(train_features, train_labels)

        if test_features is None:
            return model
        else:
            predictions = model.predict(test_features)
            return model, predictions

    def use_lgb(self, train_features, train_labels, rate_learning, n=100, test_features=None):
        """
        :param train_features: dataframe of features for training
        :param train_labels:  dataframe of labels for training
        :param rate_learning: used to select the learning rate for the model
        :param n: Optional, default is 100. Used to select the number of trees
        :param test_features: Optional, default is None. If not None, dataframe of features is used for predicting labels
        :return: model and label predictions (if test_features is not None)
        """

        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=n, learning_rate=rate_learning, class_weight=None)
        model.fit(train_features, train_labels)

        if test_features is None:
            return model
        else:
            predictions = model.predict(test_features)
            return model, predictions

    def confusion_matrix_results(self, test_labels, predictions):
        """
        :param test_labels: array of labels for testing
        :param predictions: array of predicted labels
        :return: array for confusion matrix and float of roc score
        """

        from sklearn.metrics import confusion_matrix, roc_auc_score

        conf_mtx = confusion_matrix(test_labels, predictions)

        return conf_mtx

    def model_train_and_prediction(train_dataset, test_dataset, algorithm, random_state=1, learning_rate=None, n_estimators=None, train_size=None, doTest=False):
        """
        :param train_dataset: dataframe of training data
        :param test_dataset: dataframe of testing data
        :param algorithm: indicates which model is used for training and testing
        :param random_state: Sets random state, default is 1
        :param learning_rate: Sets the learning rate for XGBoost
        :param n_estimators: Sets the number of trees
        :param train_size: Sets the percentage of rows used for training. Value must be between 0 and 1
        :param doTest: Optional, default is False which indicates the entire market_training dataset is used for training. True indicates market_training is used for training and testing
        :return: If doTest is False, an array of the predictions for market_testing dataset. If doTest is True, the model and arrays of testing features and labels and the predictions
        """

        # when we are training on the entire train_dataset
        if not doTest:  # if doTest is False
            train_features, train_labels = self.split_features_labels(train_dataset)
            train_features = train_features.values
            train_labels = train_labels.values
            test_dataset = test_dataset.values

            # use algorithm selected for training model
            if algorithm == 'xgboost':
                model = self.use_xgboost(train_features, train_labels, random_state, learning_rate, 9)
            elif algorithm == 'random forest':
                model = self.use_random_forest(train_features, train_labels, n_estimators, random_state)
            elif algorithm == 'lgb':
                model = self.use_lgb(train_features, train_labels, None)
            else:
                print('Error: value for algorithm_model should be xgboost, random forest, or lgb.')

            predictions = model.predict(test_dataset)

            return predictions

        # when we are training and testing on same dataset
        else:
            train_features, train_labels, test_features, test_labels = self.split_features_labels(train_dataset, doTest,
                                                                                             train_size, random_state)
            train_features = train_features.values
            train_labels = train_labels.values
            test_features = test_features.values
            test_labels = test_labels.values

            # use algorithm selected for training model
            if algorithm == 'xgboost':
                model, predict_responded = self.use_xgboost(train_features, train_labels, random_state, learning_rate, 9,
                                                       test_features)
            elif algorithm == 'random forest':
                model, predict_responded = self.use_random_forest(train_features, train_labels, n_estimators, random_state,
                                                             test_features)
            elif algorithm == 'lgb':
                model, predict_responded = self.use_lgb(train_features, train_labels, learning_rate, 100, test_features)
            else:
                print('Error: value for algorithm_model should be xgboost, random forest, or lgb.')

            return model, predict_responded, test_features, test_labels