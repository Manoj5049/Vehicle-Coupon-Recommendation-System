import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB, BernoulliNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle   



def data_cleaning(work_df):
    #Chaning column names
    work_df = work_df.rename(columns={'coupon': 'coupon_type','Y':'coupon_status'})
    work_df.columns
    #Changing Datatypes
    work_df=work_df.astype('category')
    work_df['temperature']=work_df['temperature'].astype('int')
    work_df['toCoupon_GEQ5min']=work_df['toCoupon_GEQ5min'].astype('int')
    work_df['toCoupon_GEQ15min']=work_df['toCoupon_GEQ15min'].astype('int')
    work_df['toCoupon_GEQ25min']=work_df['toCoupon_GEQ25min'].astype('int')
    work_df['direction_same']=work_df['direction_same'].astype('int')
    work_df['direction_opp']=work_df['direction_opp'].astype('int')
    work_df['age'] = work_df['age'].replace({'50plus':51,'below21':20})
    work_df['age']=work_df['age'].astype('int')
    work_df['coupon_status']=work_df['coupon_status'].astype('int')
    # The total null values in column car are huge so this column is not useful for analysis. So, drop the column.
    work_df.drop(columns=['car'], inplace=True)
    #change the values to meaningful values
    for ele in ['Bar','CoffeeHouse','CarryAway','RestaurantLessThan20','Restaurant20To50']:
        work_df[ele] = work_df[ele].replace({'less1':'less_than_1','1~3':'1_to_3','4~8':'4_to_8', 'gt8':'greater_than_8'})
    #Removing deplicates
    work_df=work_df.drop_duplicates()
    # Removeming those rows
    work_df=work_df[~work_df[['Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50']].isnull().all(axis=1)]
    #reset index
    work_df=work_df.reset_index(drop=True)
    #mode Imputation
    modes = work_df.mode().iloc[0]
    work_df = work_df.fillna(modes)
    work_df.isnull().sum()
    #encoding
    label_encoder = LabelEncoder()
    work_df['destination'] = label_encoder.fit_transform(work_df['destination'])
    destination_lables = dict(zip(label_encoder.inverse_transform(work_df['destination']),work_df['destination']))
    work_df['passanger'] = label_encoder.fit_transform(work_df['passanger'])
    passanger_lables = dict(zip(label_encoder.inverse_transform(work_df['passanger']),work_df['passanger']))
    work_df['weather'] = label_encoder.fit_transform(work_df['weather'])
    weather_lables = dict(zip(label_encoder.inverse_transform(work_df['weather']),work_df['weather']))
    work_df['time'] = label_encoder.fit_transform(work_df['time'])
    time_lables = dict(zip(label_encoder.inverse_transform(work_df['time']),work_df['time']))
    work_df['coupon_type'] = label_encoder.fit_transform(work_df['coupon_type'])
    coupon_type_lables = dict(zip(label_encoder.inverse_transform(work_df['coupon_type']),work_df['coupon_type']))
    work_df['gender'] = label_encoder.fit_transform(work_df['gender'])
    gender_lables = dict(zip(label_encoder.inverse_transform(work_df['gender']),work_df['gender']))
    work_df['maritalStatus'] = label_encoder.fit_transform(work_df['maritalStatus'])
    maritalStatus_lables = dict(zip(label_encoder.inverse_transform(work_df['maritalStatus']),work_df['maritalStatus']))
    work_df['has_children'] = label_encoder.fit_transform(work_df['has_children'])
    has_children_lables = dict(zip(label_encoder.inverse_transform(work_df['has_children']),work_df['has_children']))
    work_df['education'] = label_encoder.fit_transform(work_df['education'])
    education_lables = dict(zip(label_encoder.inverse_transform(work_df['education']),work_df['education']))
    work_df['occupation'] = label_encoder.fit_transform(work_df['occupation'])
    occupation_lables = dict(zip(label_encoder.inverse_transform(work_df['occupation']),work_df['occupation']))
    work_df['expiration'] = label_encoder.fit_transform(work_df['expiration'])
    expiration_lables = dict(zip(label_encoder.inverse_transform(work_df['expiration']),work_df['expiration']))
    work_df['Bar'] = work_df["Bar"].map({'never':0,'less_than_1':0,'1_to_3':1,'4_to_8':2, 'greater_than_8':3})
    Bar_lables = {'never':0,'less_than_1':0,'1_to_3':1,'4_to_8':2, 'greater_than_8':3}
    work_df['CoffeeHouse'] = work_df["CoffeeHouse"].map({'never':0,'less_than_1':0,'1_to_3':1,'4_to_8':2, 'greater_than_8':3})
    CoffeeHouse_labels = {'never':0,'less_than_1':0,'1_to_3':1,'4_to_8':2, 'greater_than_8':3}
    work_df['CarryAway'] = work_df["CarryAway"].map({'never':0,'less_than_1':0,'1_to_3':1,'4_to_8':2, 'greater_than_8':3})
    CarryAway_lables = {'never':0,'less_than_1':0,'1_to_3':1,'4_to_8':2, 'greater_than_8':3}
    work_df['RestaurantLessThan20'] = work_df["RestaurantLessThan20"].map({'never':0,'less_than_1':0,'1_to_3':1,'4_to_8':2, 'greater_than_8':3})
    RestaurantLessThan20_lables = {'never':0,'less_than_1':0,'1_to_3':1,'4_to_8':2, 'greater_than_8':3}
    work_df['Restaurant20To50'] = work_df["Restaurant20To50"].map({'never':0,'less_than_1':0,'1_to_3':1,'4_to_8':2, 'greater_than_8':3})
    Restaurant20To50_labels = {'never':0,'less_than_1':0,'1_to_3':1,'4_to_8':2, 'greater_than_8':3}
    work_df['income'] = work_df["income"].map({'$37500 - $49999':3,'$62500 - $74999':5, '$12500 - $24999':1, '$75000 - $87499':6,
    '$50000 - $62499':4, '$25000 - $37499':2,'$100000 or More':8, '$87500 - $99999':7,'Less than $12500':0})
    income_labels = {'$37500 - $49999':3,'$62500 - $74999':5, '$12500 - $24999':1, '$75000 - $87499':6,'$50000 - $62499':4, '$25000 - $37499':2,'$100000 or More':8, '$87500 - $99999':7,'Less than $12500':0}
    work_df['income']=work_df['income'].astype(int)
    ##9 Feature Engineering
    #from the above feature 'toCoupon_GEQ5min' as alone is unimportant,But logically is important and also 
    #toCoupon_GEQ15min and toCoupon_GEQ25min. lets combine these three give rank accordingly(less distance more possibility to use)
    work_df['toCoupon_GEQ']=4-(work_df['toCoupon_GEQ5min']+work_df['toCoupon_GEQ15min']+work_df['toCoupon_GEQ25min'])
    work_df.drop(columns=['toCoupon_GEQ5min','toCoupon_GEQ15min','toCoupon_GEQ25min'], inplace=True)
    Labels = [destination_lables, passanger_lables, weather_lables, time_lables, coupon_type_lables , expiration_lables,'', maritalStatus_lables, has_children_lables, education_lables, income_labels , Bar_lables,CoffeeHouse_labels, CarryAway_lables, RestaurantLessThan20_lables, Restaurant20To50_labels,'']
    return work_df, Labels

def model(work_df):
    # Test Train Split
    X=work_df.drop(columns=['coupon_status','direction_opp','occupation','direction_same','gender','temperature']) 
    Y=work_df['coupon_status']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=108)
    
    # 1)Logestic Regression 
    log_model = LogisticRegression(max_iter=1000,C=0.1)
    log_model.fit(X_train, Y_train)

    
    #2)Random forest
    rf_classifier = RandomForestClassifier(n_estimators=500, random_state=42,max_depth=20)
    rf_classifier.fit(X_train, Y_train)
   

    #3) Gradient Boosting Machines
    gbm_classifier = GradientBoostingClassifier(n_estimators=300, random_state=42,learning_rate= 0.2, max_depth= 5)
    gbm_classifier.fit(X_train, Y_train)
    
    #4)KNN
    knn_classifier = KNeighborsClassifier(n_neighbors=7,metric='manhattan', weights='distance')
    knn_classifier.fit(X_train, Y_train)
    
    ##5)Navie Bayes - No hypertuning BUT categorical NB has good results
    nb_classifier = CategoricalNB()
    nb_classifier.fit(X_train, Y_train)

    #6)SVM 
    svm_classifier = SVC(C= 1, gamma= 0.1, kernel='rbf')
    svm_classifier.fit(X_train, Y_train)

    
    #7)XGB 
    xgb_classifier = XGBClassifier(learning_rate= 0.1, max_depth= 5, n_estimators= 500)
    xgb_classifier.fit(X_train, Y_train)

    return([log_model,rf_classifier,gbm_classifier,knn_classifier,nb_classifier,svm_classifier,xgb_classifier],X_train)


data=pd.read_csv("C:\\Users\\palad\\Desktop\\in+vehicle+coupon+recommendation\\in-vehicle-coupon-recommendation.csv")
work_df=data.copy()
cleaned_df,Labels=data_cleaning(work_df)
models,train_df=model(cleaned_df)
filename = 'train_df.sav'
pickle.dump(train_df, open(filename, 'wb'))
filename = 'models.sav'
pickle.dump(models, open(filename, 'wb'))
filename = 'labels.sav'
pickle.dump(Labels, open(filename, 'wb'))
