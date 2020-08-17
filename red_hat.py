#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import cross_validate, train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
from scipy.stats import chi2_contingency, pointbiserialr
from category_encoders import TargetEncoder



#### IMPORTING FILES

def create_dir(dirname):
    """
    The function takes in the path of a desired directory.
    
    It either creates the directory if it doesn't exist or prints out the information that the directory exists.
    """
    if not os.path.exists(dirname):
        print(f"Creating {dirname} directory")
        os.mkdir(dirname)
    else:
        print(f"{dirname} directory already exists")
        pass

create_dir('./data')
create_dir('./model')



def import_csv(filename, path = './data'):
#the aim of this function is to import csv files as dataframes
    return pd.read_csv(os.path.join(path, filename))

train_activity = import_csv('act_train.csv')
people = import_csv('people.csv')



#### DATA CLEANING

def _process_date(dataframe, date_col):
    """
    A private function which preprocesses datetime information.
    Input:
    - dataframe
    - name of the date column (object)
    
    Output:
    - dataframe with new date columns: month, year, weekend_flg
    """
    df = dataframe.copy()
    df['date'] = pd.to_datetime(df[date_col])
    df['month']=df['date'].dt.month
    df['year']=df['date'].dt.year
    df['weekend_flg'] = (df['date'].dt.weekday >= 5).astype(int)
    df.drop(['date'], inplace=True, axis=1)
    return df



def clean_people(original_df):
    """
    The aim of this function is to prepare `people` df by unifying types of data.
    The function takes in a dataframe (specifically `people`) and returns a copy of the given dataframe, but with converted data types (all ints).
    """
    df = _process_date(original_df, 'date')
    
    for col in list(df.select_dtypes(include='object').columns):
        if col.startswith("char_") or col.startswith("group_"):
            try:
                df[col] = (df[col].apply(lambda x: x.split(" ")[1]).astype("float64")).astype('int64')
                print(f"{col} converted to int")
            except AttributeError:
                print(f"Can't convert {col} to int")

        elif col.startswith("people_"):
            try:
                df[col] = (df[col].apply(lambda x: x.split("_")[1]).astype("float64")).astype('int64')
                print(f'{col} converted to int')
            except AttributeError:
                print(f"Can't convert {col} to int")
                
    for col in list(df.select_dtypes(include=['bool', 'float64']).columns):
        try:
            df[col] = df[col].astype("int64")
            print(f"{col} converted to int")
        except AttributeError:
            print(f"Can't convert {col} to int")
    return df

people_df = clean_people(people)
del people



def clean_activity(original_df):
    """
    The aim of this function is to prepare `activity` df by unifying types of data.
    The function takes in a dataframe (specifically `activity`) and returns this dataframe, but with converted data types.
    """ 

    df = _process_date(original_df, 'date')
                               
    for col in list(df.select_dtypes(include='object').columns):
        if col.endswith("_id"):
            if col.startswith("activity"):
                try:
                    df[f"{col}_prefix"] = (df[col].apply(lambda x: x.split("_")[0][-1]).astype("float64")).astype("int64")
                    print(f"{col}_prefix created")
                except AttributeError:
                    print(f"Can't create {col}_prefix")
                try:
                    df[col] = (df[col].apply(lambda x: x.split("_")[1]).astype("float64")).astype("int64")
                    print(f"{col} converted to int")
                except AttributeError:
                    print(f"Can't convert {col} to int")              
            elif col.startswith("people"):
                try:
                    df[col] = (df[col].apply(lambda x: x.split("_")[1]).astype("float64")).astype("int64")
                    print(f"{col} converted to int")
                except AttributeError:
                    print(f"Can't convert {col} to int")
        else:
            df[col]= df[col].fillna('type -1')
            try:
                df[col] = (df[col].apply(lambda x: x.split(" ")[1]).astype("float64")).astype('int64')
                print(f"{col} converted to int")
            except AttributeError:
                print(f"Can't convert {col} to int")
                
    for col in list(df.select_dtypes(include=['bool', 'float64']).columns):
        try:
            df[col] = df[col].astype("int64")
            print(f"{col} converted to int")
        except AttributeError:
            print(f"Can't convert {col} to int")
    df.loc[:,'activity_index'] = df[['activity_id_prefix', 'activity_id']].apply(tuple, axis=1)
    return df

train_activity_df= clean_activity(train_activity)
del train_activity



# new pandas 1.0 feature - convert_dtypes(), to handle missing values
red_hat = pd.merge(people_df, train_activity_df, how = 'left', on = 'people_id', suffixes = ('_pep', '_act')).convert_dtypes()

#deleting records where outcome or activity_id are NaNs, because they are useless in case of modeling
red_hat = red_hat[(pd.isna(red_hat['activity_id'])== False) & (pd.isna(red_hat['outcome'])== False)]

train_set, test_set = train_test_split(red_hat, test_size = 0.22, random_state = 42, stratify = red_hat['outcome'])

train_set = train_set.set_index('activity_index')
test_set = test_set.set_index('activity_index')

del people_df
del train_activity_df



#### FEATURE SELECTION

index_cols = ['activity_id_prefix', 'activity_id']
cat_cols = ['char_10_act' , 'group_1']
target = ['outcome']
continuous_cols = ['char_38']

#categorical variables to check correlation between them and target
cols_corr = [x for x in list(train_set.columns) if x not in (target+continuous_cols+index_cols)]



def cramers_corrected_stat(cols_to_check, df, target, thresh):
    """ 
    The aim of the function is to calculate the corrected version of Cramer's V to find the level of association between categorical variables.
    
    The function takes in:
    - a dataframe with categorical variables, 
    - a list of categorical variables to check, 
    - the name of the column with target,
    - a threshold for Cramer's V values from which strong association will be assumed.
    
    The result of the function is a list of variables which are strongly associated with the target, according to the Cramer's V values.
    """
    
    cols_to_drop = []
    
    for col in cols_to_check:

        confusion_matrix = pd.crosstab(df[col],df[target])
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2/n
        r,c = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((c-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        ccorr = c - ((c-1)**2)/(n-1)
        cramers_v = np.sqrt(phi2corr / min( (ccorr-1), (rcorr-1)))
        
        if cramers_v > thresh:
            print(f'Target and {col} are associated: {round(cramers_v,2)}')
            cols_to_drop.append(col)
        else:
            pass
    
    return cols_to_drop

cramer_cols = cramers_corrected_stat(cols_corr, train_set, 'outcome', 0.5)



def point_biserial_correlation(df, contin_cols, target, thresh):
    """
    The aim of the function is to calculate a point biserial correlation coefficient and the associated p-value.
    
    The function takes in:
    - a dataframe with variables for which we want to test correlation
    - a list of continuous variable
    - a binary variable (target)
    - a threshold for correlation coeficient from which strong correlation will be assumed.
    
    The result of the function is a list of variables with high correlation.
    """
    
    cols_to_drop = []
    
    for col in contin_cols:
    
        corr = pointbiserialr(df[col],df[target])[0]
        p = pointbiserialr(df[col],df[target])[1]
    
        if (p<=0.5):
            if (abs(corr)>thresh):
                print(f"{col} is correlated with the target: {round(corr,2)}")
                cols_to_drop.append(col)
            else:
                print(f"{col} with low correlation")
        else:
            print(f"No statistically significant correlation")
    
    return cols_to_drop

point_biserial_cols = point_biserial_correlation(train_set, continuous_cols, 'outcome', 0.5)



def cramers_corr_features(df, cat_cols, thresh):
    """
    The purpose of this function is to calculate the corrected version of Cramer's V for all the pairs of categorical variables to identify intercorrelated
    variables.
    
    The function takes in:
    - a dataframe with features (X)
    - a list of categorical variables
    - a threshold for classifying variables as correlated
    
    The output of this function is a list of intercorrelated variables.
    """
    
    df_corr = pd.DataFrame(index = cat_cols, columns = cat_cols, data = 0)
    #to avoid duplicates of the pairs of variables, only combinations from the triangle will be taken
    df_pairs = pd.DataFrame(df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(np.bool)).stack()).reset_index().rename(columns={'level_0':'var1','level_1':'var2'}).iloc[:,0:2]    
    combinations = []

    #appending to the combinations list unique pairs of variables
    for x in range(df_pairs.shape[0]):
        combinations.append(list(df_pairs.iloc[x,0:2]))
        
    
    corr_features = []

    for pair in combinations:
        if (pair[0] not in corr_features) & (pair[1] not in corr_features):
            #calculating cramer's v for variables that are not yet correlated with previously checked variables
            confusion_matrix = pd.crosstab(df[pair[0]],df[pair[1]])
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2/n
            r,c = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((c-1)*(r-1))/(n-1))    
            rcorr = r - ((r-1)**2)/(n-1)
            ccorr = c - ((c-1)**2)/(n-1)
            cramers_v = np.sqrt(phi2corr / min( (ccorr-1), (rcorr-1)))

            if cramers_v > thresh:
                print(f'{pair[0]} and {pair[1]} are associated: {round(cramers_v,2)}, dropping {pair[0]}')
                # Only one of the correlated variables will be added to the list of correlated variables.
                # The variable is picked randomly as additional calculations on such a big volume of data is to costly.
                corr_features.append(pair[0])
            else:
                pass
            
    return corr_features

correlated_features = cramers_corr_features(train_set, cols_corr, 0.8)



def frequency_check(df, cat_cols, thresh):
    """
    The aim of the function is to find features with categories of high frequency.
    
    The function takes in:
    - a dataframe with features
    - list of categorical columns
    - threshold for high frequency
    
    The output of the function is a list of variables with categories of high frequency.
    """
    
    high_freq = []
    df_cat = df[cat_cols]
    for col in list(df_cat.columns):
        max_freq = df_cat[col].value_counts(normalize=True).sort_values(ascending=False).max()
        if max_freq > thresh:
            high_freq.append(col)
            
    return high_freq

high_frequency_cols = frequency_check(train_set, cols_corr, 0.8)



X_train = train_set.drop('outcome', axis=1)
y_train = train_set['outcome']

X_test = test_set.drop('outcome', axis=1)
y_test = test_set['outcome']

del train_set
del test_set



cols_to_drop = list(set(index_cols+cat_cols+cramer_cols+point_biserial_cols+correlated_features+high_frequency_cols))
modelling_cols = [x for x in X_train.columns if x not in cols_to_drop]



categories ={}
#find the number of categories for each variable
for cat in modelling_cols:
    if cat != 'outcome':
        categories[cat]= len((list(X_train[cat].unique())))



def define_cols_to_encode(cat_dict, thresh):
    """
    Defining the lists of columns to encode depending on the number of categories per feature.
    
    The function takes in:
    - a dictionary with name of columns as keys and number of categories as values
    - (min. number -1) of categories to include in frequency encoding and at the same time max. number of categories to include in binary encoding 
    """                       
    binary_cat = list({k for k, v in cat_dict.items() if v == 2})
    little_cat = list({k for k, v in cat_dict.items() if v in range(3,10)})
    big_cat = list({k for k, v in cat_dict.items() if v >= 10})
                             
    return little_cat, big_cat, binary_cat                   

thresh = 10
little_cat, big_cat, binary_cat = define_cols_to_encode(categories, thresh)



#### PREPROCESSING PIPELINE

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Extracts only a given list of columns and returns a filtered dataframe.
    """
    def __init__(self, feature_names):
        self._feature_names = feature_names 
    
    def fit(self, X, y = None):
        return self 
    
    def transform(self, X, y = None):
        return X[self._feature_names] 



class ValueImputer(BaseEstimator, TransformerMixin):
    """
    Fills missing values with a constant and returns a dataframe with imputed values
    """
    def __init__(self, impute_value):
        self.impute_value = impute_value
    
    def fit(self, X, y = None):
        return self
        
    def transform(self, X, y = None):   
        return X.fillna(self.impute_value)

    def fit_transform(self, X, y = None):
        return X.fillna(self.impute_value)



class Encoder01(BaseEstimator, TransformerMixin):
    """
    Encodes categorical variables using their frequencies.
    """
    def __init__(self, binary_cols):
        """
        Freq_cols is a list of columns which will be encoded using the Encoder01
        """
        self.binary_cols = binary_cols
    
    def fit(self, X, y = None):
        """
        The fit method takes in a DataFrame with features (X) and a numpy array with the target variable (y).
        
        It creates a dictionary, where keys are names of features and values are dictionaries (zipped uniques&zero_ones).
        In the zipped dictionary keys are the names of categories represented by a specific feature and the values are the new binary values : 0 or 1.
        """
        self.maps ={}
        for col in self.binary_cols:
            self.maps[col] = []
            uniques = sorted(list(X[col].unique()))
            zero_ones = [0,1]
            self.maps[col]  = dict(zip(uniques, zero_ones)) 
        return self
        
    def transform(self, X, y = None):
        """
        The transform method takes in a DataFrame with features (X) and a numpy array with the target variable (y).
        
        The transform method replaces the names of categories with zeros or ones (using values stored in `map` dictionary).
        If a given category is not in the dictionary, it is encoded with "-1".
        """
        
        for var in self.maps.keys():
            try:
                X[var] = X[var].apply(lambda x: self.maps[var][x])
            except KeyError:
                print("Missing key for test set")
                X[var] = X[var].apply(lambda x: -1)
        return X
        
        
    def fit_transform(self, X, y = None):
        """
        Combines the above mentioned fit and transform methods.
        """
        return self.fit(X, y).transform(X, y)



class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical variables using their frequencies.
    """
    def __init__(self, freq_cols):
        """
        Freq_cols is a list of columns which will be encoded using the FrequencyEncoder
        """
        self.freq_cols = freq_cols
    
    def fit(self, X, y = None):
        """
        The fit method takes in a DataFrame with features (X) and a numpy array with the target variable (y).
        
        It creates a dictionary, where keys are names of features and values are dictionaries (zipped uniques&frequencies).
        In the zipped dictionary keys are the names of categories represented by a specific feature and the values are their frequencies of occurance in the set.
        """
        self.maps ={}
        for col in self.freq_cols:
            self.maps[col] = {}
            uniques = list(X[col].unique())
            frequencies = list(X.groupby(col).size()/ len(X))
            self.maps[col]  = dict(zip(uniques, [round(x,3) for x in frequencies])) 
        return self
        
    def transform(self, X, y = None):
        """
        The transform method takes in a DataFrame with features (X) and a numpy array with the target variable (y).
        
        The transform method replaces the names of categories with the frequencies of those categories in the dataset (using values stored in `map` dictionary).
        If a given category is not in the dictionary, it is encoded with "-1".
        """
        for var in self.maps.keys():
            try:
                X[var] = X[var].apply(lambda x: self.maps[var][x])
            except KeyError:
                print("Missing key for test set")
                X[var] = X[var].apply(lambda x: -1)
        return X

    def fit_transform(self, X, y = None):
        """
        Combines the above mentioned fit and transform methods.
        """
        return self.fit(X, y).transform(X, y)



cat_pipeline = Pipeline([
        ('column_selector', ColumnSelector(modelling_cols)),
        ('imputer', ValueImputer("-1")),
        ('binary_encoder', Encoder01(binary_cat)),
        ('frequency_encoder', FrequencyEncoder(little_cat)),
        ('target_encoder', TargetEncoder(cols = big_cat, smoothing = 0.8))
    ])

#converting some variables to string as the encoder works only on objects
X_train[little_cat + big_cat] = X_train[little_cat + big_cat].astype('str')
X_test[little_cat + big_cat] = X_test[little_cat + big_cat].astype('str')

X_train_t = cat_pipeline.fit_transform(X_train, y_train)
del X_train

X_test_t = cat_pipeline.transform(X_test)
del X_test

y_train = y_train.astype('int')
y_test = y_test.astype('int')



#### MODELLING

#specifying the random seed to enable reproducibility of the predictions
random_seed = 42

def make_predictions(classifier, X, y):
    """
    The function takes in a fitted model, dataframe with features and a numpy array with the outcomes.
    
    It returns a numpy array with lists containing predicted probabilites of the target = 0 and target = 1.
    """
    pred = cross_val_predict(classifier, X, y, cv =5, method = 'predict_proba', verbose = False)
    return pred



def evaluate_scores(y, y_pred, thresh=0.5):
    """
    The function takes in:
    - a numpy array with the true target values, 
    - a numpy array with predicted probabilities of the target 
    - a probability threshold for classification.
    
    It returns a dataframe with two columns: names of metrics (accuracy, precision, recall, f1, roc auc score) and scores.
    """
    #creating a numpy array with predicted values of the target (0 or 1) based on the probability threshold
    y_pred_vals = (y_pred[:,1] >= thresh).astype('int')
    
    # metrics
    acc = accuracy_score(y, y_pred_vals)
    prec = precision_score(y, y_pred_vals)
    rec = recall_score(y, y_pred_vals)
    f1 = f1_score(y, y_pred_vals)
    roc = roc_auc_score(y, y_pred_vals)
    
    metrics = pd.DataFrame({'metrics': ['accuracy', 'precision','recall','f1','roc_auc'],
                            'scores': [acc, prec, rec, f1, roc]})
    
    return metrics



def evaluate_cmatrix(y, y_pred, thresh = 0.5):
    """
    The function takes in:
    - a numpy array with the true target values, 
    - a numpy array with predicted probabilities of the target 
    - a probability threshold for classification.
    
    It returns a dataframe with the confusion matrix, where rows represten the actual class and columns - predicted class.
    """
    #creating a numpy array with predicted values of the target (0 or 1) based on the probability threshold    
    y_pred_vals = (y_pred[:,1] >= thresh).astype('int')
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred_vals).ravel()
    
    # PN - predicted negative, PP - predicted positive, TN - true negative, TP - true positive
    cmatrix = pd.DataFrame({'PN': [tn, fn],
                            'PP':[fp, tp]},
                            index = ['TN','TP'])
    
    return cmatrix



#### FINDING BEST PARAMS

scoring = 'f1'

def rf_best_params(X, y, random_seed, scoring,
                         max_features, max_depth, min_samples_split, 
                         min_samples_leaf, n_estimators):
    
    rf_def = RandomForestClassifier(random_state = random_seed)
    
    grid = {'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'n_estimators': n_estimators}
    
    rf_rsearch = RandomizedSearchCV(rf_def, param_distributions=grid, n_iter= 3, n_jobs=-1, cv = 3)
    
    rf_rsearch.fit(X, y)
    
    return rf_rsearch.best_params_, rf_rsearch.best_score_



## first iteration
max_features = ['auto','log2']
max_depth = range(20, 26, 2)
min_samples_split = range(10,16,2)
min_samples_leaf = range(10, 18, 3)
n_estimators = [100]    

best_params, best_score = rf_best_params(X_train_t, y_train, random_seed, scoring, max_features, max_depth, min_samples_split, min_samples_leaf, n_estimators)



## second iteration
max_features = [best_params['max_features']]
max_depth = range(best_params['max_depth']-1, best_params['max_depth']+3)
min_samples_split = range(best_params['min_samples_split']-1, best_params['min_samples_split']+2)
min_samples_leaf = range(best_params['min_samples_leaf']-2, best_params['min_samples_leaf']+2)
n_estimators = [best_params['n_estimators']]

best_params1, best_score1 = rf_best_params(X_train_t, y_train, random_seed, scoring, max_features, max_depth, min_samples_split, min_samples_leaf, n_estimators)

if best_score1 > best_score:
    best_score = best_score1
    best_params = best_params1



#third iteration
rf = RandomForestClassifier(random_state = random_seed, n_estimators = 200)

##calculating mean cross-validated score of the estimator to later compare it to the scores after hyperparameter tuning
best_score2 = np.mean(cross_val_score(rf, X_train_t, y_train, cv=3, scoring = scoring))
best_params2 = rf.get_params()

if best_score2 > best_score:
    best_score = best_score2
    best_params = best_params2



## training the final model with the best params
rf_final = RandomForestClassifier(**best_params)
rf_final.fit(X_train_t, y_train)

y_train_pred  = make_predictions(rf_final, X_train_t, y_train)
y_test_pred  = make_predictions(rf_final, X_test_t, y_test)

metrics_train = evaluate_scores(y_train, y_train_pred)
metrics_test = evaluate_scores(y_test, y_test_pred)

cm_train = evaluate_cmatrix(y_train, y_train_pred)
cm_test = evaluate_cmatrix(y_test, y_test_pred)



#### SAVING MODEL AND BEST PARAMS

with open('./model/best_params_dict.pickle', 'wb') as handle:
    pickle.dump(best_params, handle, protocol = pickle.HIGHEST_PROTOCOL)

pickle.dump(rf_final, open('./model/best_model_rf.sav', 'wb'))

## printing the results of the model (metrics and confusion matrix)
print(metrics_train)
print(metrics_test)
print(cm_train)
print(cm_test)