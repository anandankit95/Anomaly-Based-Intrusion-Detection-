
# coding: utf-8

# # Modelling Intrusion Detection: Analysis of a Feature Selection Mechanism
# 
# ## Method Description
# 
# ### Step 1: Data preprocessing:
# All features are made numerical using one-Hot-encoding. The features are scaled to avoid features with large values that may weigh too much in the results.
# 
# ### Step 2: Feature Selection:
# Eliminate redundant and irrelevant data by selecting a subset of relevant features that fully represents the given problem.
# Univariate feature selection with ANOVA F-test. This analyzes each feature individually to detemine the strength of the relationship between the feature and labels. Using SecondPercentile method (sklearn.feature_selection) to select features based on percentile of the highest scores. 
# When this subset is found: Recursive Feature Elimination (RFE) is applied.
# 
# ### Step 4: Build the model:
# Decision tree model is built.
# 
# ### Step 5: Prediction & Evaluation (validation):
# Using the test data to make predictions of the model.
# Multiple scores are considered such as:accuracy score, recall, f-measure, confusion matrix.
# perform a 10-fold cross-validation.





import pandas as pd
import numpy as np
import sys
import sklearn
import csv
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Dropout 
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import os
import operator
#print(pd.__version__)
#print(np.__version__)
#print(sys.version)
#print(sklearn.__version__)


# ## Load the Dataset



# attach the column names to the dataset
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

# KDDTrain+_2.csv & KDDTest+_2.csv are the datafiles without the last column about the difficulty score
# these have already been removed.
df = pd.read_csv("KDDTrain+_2.csv", header=None, names = col_names)
df_test = pd.read_csv("KDDTest+_2.csv", header=None, names = col_names)
#print(type(df_test))

# shape, this gives the dimensions of the dataset
#print('Dimensions of the Training set:',df.shape)
#print('Dimensions of the Test set:',df_test.shape)


# ## Sample view of the training dataset



# first five rows
df.head(10)


# ## Statistical Summary



df.describe()


# ## Label Distribution of Training and Test set



#print('Label distribution Training set:')
#print(df['label'].value_counts())
#print()
#print('Label distribution Test set:')
#print(df_test['label'].value_counts())


# # Step 1: Data preprocessing:
# One-Hot-Encoding (one-of-K) is used to to transform all categorical features into binary features. 
# Requirement for One-Hot-encoding:
# "The input to this transformer should be a matrix of integers, denoting the values taken on by categorical (discrete) features. The output will be a sparse matrix where each column corresponds to one possible value of one feature. It is assumed that input features take on values in the range [0, n_values)."
# 
# Therefore the features first need to be transformed with LabelEncoder, to transform every category to a number.

# ## Identify categorical features



# colums that are categorical and not binary yet: protocol_type (column 2), service (column 3), flag (column 4).
# explore categorical features
#print('Training set:')
for col_name in df.columns:
    if df[col_name].dtypes == 'object' :
        unique_cat = len(df[col_name].unique())
        #print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

#see how distributed the feature service is, it is evenly distributed and therefore we need to make dummies for all.
#print()
#print('Distribution of categories in service:')
#print(df['service'].value_counts().sort_values(ascending=False).head())




# Test set
#print('Test set:')
for col_name in df_test.columns:
    if df_test[col_name].dtypes == 'object' :
        unique_cat = len(df_test[col_name].unique())
        #print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))


# ### Conclusion: Need to make dummies for all categories as the distribution is fairly even. In total: 3+70+11=84 dummies.
# ### Comparing the results shows that the Test set has fewer categories (6), these need to be added as empty columns.

# # LabelEncoder

# ### Insert categorical features into a 2D numpy array



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
categorical_columns=['protocol_type', 'service', 'flag']
# insert code to get a list of categorical columns into a variable, categorical_columns
categorical_columns=['protocol_type', 'service', 'flag'] 
 # Get the categorical values into a 2D numpy array
df_categorical_values = df[categorical_columns]
testdf_categorical_values = df_test[categorical_columns]
df_categorical_values.head()


# ### Make column names for dummies



# protocol type
unique_protocol=sorted(df.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2=[string1 + x for x in unique_protocol]
# service
unique_service=sorted(df.service.unique())
string2 = 'service_'
unique_service2=[string2 + x for x in unique_service]
# flag
unique_flag=sorted(df.flag.unique())
string3 = 'flag_'
unique_flag2=[string3 + x for x in unique_flag]
# put together
dumcols=unique_protocol2 + unique_service2 + unique_flag2
#print(dumcols)
    
#UP TILL HERE -- MADE 84 DUMMIES for categoricals

#do same for test set
unique_service_test=sorted(df_test.service.unique())
unique_service2_test=[string2 + x for x in unique_service_test]
testdumcols=unique_protocol2 + unique_service2_test + unique_flag2


# ## Transform categorical features into numbers using LabelEncoder()


df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)
#print(df_categorical_values_enc.head())
# test set
testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)


# # One-Hot-Encoding



enc = OneHotEncoder()
df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)
# test set
testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdumcols)

df_cat_data.head()


# ### Add 6 missing categories from train set to test set



trainservice=df['service'].tolist()
testservice= df_test['service'].tolist()
difference=list(set(trainservice) - set(testservice))
string = 'service_'
difference=[string + x for x in difference]
difference




for col in difference:
    testdf_cat_data[col] = 0

testdf_cat_data.shape


# ## Join encoded categorical dataframe with the non-categorical dataframe



newdf=df.join(df_cat_data)
newdf.drop('flag', axis=1, inplace=True)
newdf.drop('protocol_type', axis=1, inplace=True)
newdf.drop('service', axis=1, inplace=True)
# test data
newdf_test=df_test.join(testdf_cat_data)
newdf_test.drop('flag', axis=1, inplace=True)
newdf_test.drop('protocol_type', axis=1, inplace=True)
newdf_test.drop('service', axis=1, inplace=True)
#print(newdf.shape)
#print(newdf_test.shape)

#Here Obtained 123 features -- Previous 42 Features + Current Encoded 84 Features - 3 features (flag, protocol, service)


# # Split Dataset into 4 datasets for every attack category
# ## Rename every attack label: 0=normal, 1=DoS, 2=Probe, 3=R2L and 4=U2R.
# ## Replace labels column with new labels column
# ## Make new datasets
# 



# take label column
labeldf=newdf['label']
labeldf_test=newdf_test['label']
# change the label column
newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
# put the new label column back
newdf['label'] = newlabeldf
newdf_test['label'] = newlabeldf_test
#print(newdf['label'].head())




to_drop_DoS = [2,3,4]
to_drop_Probe = [1,3,4]
to_drop_R2L = [1,2,4]
to_drop_U2R = [1,2,3]
DoS_df=newdf[~newdf['label'].isin(to_drop_DoS)];
Probe_df=newdf[~newdf['label'].isin(to_drop_Probe)];
R2L_df=newdf[~newdf['label'].isin(to_drop_R2L)];
U2R_df=newdf[~newdf['label'].isin(to_drop_U2R)];

#test
DoS_df_test=newdf_test[~newdf_test['label'].isin(to_drop_DoS)];
Probe_df_test=newdf_test[~newdf_test['label'].isin(to_drop_Probe)];
R2L_df_test=newdf_test[~newdf_test['label'].isin(to_drop_R2L)];
U2R_df_test=newdf_test[~newdf_test['label'].isin(to_drop_U2R)];
#print('Train:')
#print('Dimensions of DoS:' ,DoS_df.shape)
#print('Dimensions of Probe:' ,Probe_df.shape)
#print('Dimensions of R2L:' ,R2L_df.shape)
#print('Dimensions of U2R:' ,U2R_df.shape)
#print('Test:')
#print('Dimensions of DoS:' ,DoS_df_test.shape)
#print('Dimensions of Probe:' ,Probe_df_test.shape)
#print('Dimensions of R2L:' ,R2L_df_test.shape)
#print('Dimensions of U2R:' ,U2R_df_test.shape)


# # Step 2: Feature Scaling:



# Split dataframes into X & Y
# assign X as a dataframe of feautures and Y as a series of outcome variables
X_DoS = DoS_df.drop('label',1)
Y_DoS = DoS_df.label
X_Probe = Probe_df.drop('label',1)
Y_Probe = Probe_df.label
X_R2L = R2L_df.drop('label',1)
Y_R2L = R2L_df.label
X_U2R = U2R_df.drop('label',1)
Y_U2R = U2R_df.label
# test set
X_DoS_test = DoS_df_test.drop('label',1)
Y_DoS_test = DoS_df_test.label
X_Probe_test = Probe_df_test.drop('label',1)
Y_Probe_test = Probe_df_test.label
X_R2L_test = R2L_df_test.drop('label',1)
Y_R2L_test = R2L_df_test.label
X_U2R_test = U2R_df_test.drop('label',1)
Y_U2R_test = U2R_df_test.label


# ### Save a list of feature names for later use (it is the same for every attack category). Column names are dropped at this stage.



colNames=list(X_DoS)
colNames_test=list(X_DoS_test)


# ## Use StandardScaler() to scale the dataframes



from sklearn import preprocessing
scaler1 = preprocessing.StandardScaler().fit(X_DoS)
X_DoS=scaler1.transform(X_DoS) 
scaler2 = preprocessing.StandardScaler().fit(X_Probe)
X_Probe=scaler2.transform(X_Probe) 
scaler3 = preprocessing.StandardScaler().fit(X_R2L)
X_R2L=scaler3.transform(X_R2L) 
scaler4 = preprocessing.StandardScaler().fit(X_U2R)
X_U2R=scaler4.transform(X_U2R) 
# test data
scaler5 = preprocessing.StandardScaler().fit(X_DoS_test)
X_DoS_test=scaler5.transform(X_DoS_test) 
scaler6 = preprocessing.StandardScaler().fit(X_Probe_test)
X_Probe_test=scaler6.transform(X_Probe_test) 
scaler7 = preprocessing.StandardScaler().fit(X_R2L_test)
X_R2L_test=scaler7.transform(X_R2L_test) 
scaler8 = preprocessing.StandardScaler().fit(X_U2R_test)
X_U2R_test=scaler8.transform(X_U2R_test)


"""
#Scale a single column into range
X_dos_test_instance=X_DoS_test.iloc[4,:].values
X_dos_test_instance=(X_dos_test_instance - np.mean(X_dos_test_instance)) / np.std(X_dos_test_instance)

#Make predictions using the model
"""

# ### Check that the Standard Deviation is 1



#print(X_DoS.std(axis=0))




X_Probe.std(axis=0);
X_R2L.std(axis=0);
X_U2R.std(axis=0);


# # Step 3: Feature Selection:

# # 1. Univariate Feature Selection using ANOVA F-test


"""
#univariate feature selection with ANOVA F-test. using secondPercentile method, then RFE
#Scikit-learn exposes feature selection routines as objects that implement the transform method
#SelectPercentile: removes all but a user-specified highest scoring percentage of features
#f_classif: ANOVA F-value between label/feature for classification tasks.
from sklearn.feature_selection import SelectPercentile, f_classif
np.seterr(divide='ignore', invalid='ignore');
selector=SelectPercentile(f_classif, percentile=10)
X_newDoS = selector.fit_transform(X_DoS,Y_DoS)
X_newDoS.shape


# ### Get the features that were selected: DoS

true=selector.get_support()
newcolindex_DoS=[i for i, x in enumerate(true) if x]
newcolname_DoS=list( colNames[i] for i in newcolindex_DoS )
newcolname_DoS




X_newProbe = selector.fit_transform(X_Probe,Y_Probe)
X_newProbe.shape


# ### Get the features that were selected: Probe



true=selector.get_support()
newcolindex_Probe=[i for i, x in enumerate(true) if x]
newcolname_Probe=list( colNames[i] for i in newcolindex_Probe )
newcolname_Probe




X_newR2L = selector.fit_transform(X_R2L,Y_R2L)
X_newR2L.shape


# ### Get the features that were selected: R2L



true=selector.get_support()
newcolindex_R2L=[i for i, x in enumerate(true) if x]
newcolname_R2L=list( colNames[i] for i in newcolindex_R2L)
newcolname_R2L




X_newU2R = selector.fit_transform(X_U2R,Y_U2R)
X_newU2R.shape


# ### Get the features that were selected: U2R



true=selector.get_support()
newcolindex_U2R=[i for i, x in enumerate(true) if x]
newcolname_U2R=list( colNames[i] for i in newcolindex_U2R)
newcolname_U2R


# # Summary of features selected by Univariate Feature Selection



print('Features selected for DoS:',newcolname_DoS)
print()
print('Features selected for Probe:',newcolname_Probe)
print()
print('Features selected for R2L:',newcolname_R2L)
print()
print('Features selected for U2R:',newcolname_U2R)


# ## The authors state that "After obtaining the adequate number of features during the univariate selection process, a recursive feature elimination (RFE) was operated with the number of features passed as parameter to identify the features selected". This either implies that RFE is only used for obtaining the features previously selected but also obtaining the rank. This use of RFE is however very redundant as the features selected can be obtained in another way (Done in this project). One can also not say that the features were selected by RFE, as it was not used for this. The quote could however also imply that only the number 13 from univariate feature selection was used. RFE is then used for feature selection trying to find the best 13 features. With this use of RFE one can actually say that it was used for feature selection. However the authors obtained different numbers of features for every attack category, 12 for DoS, 15 for Probe, 13 for R2L and 11 for U2R. This concludes that it is not clear what mechanism is used for feature selection. 
# 
# ## To procede with the data mining, the second option is considered as this uses RFE. From now on the number of features for every attack category is 13.

# # 2. Recursive Feature Elimination for feature ranking (Option 1: get importance from previous selected)


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
# Create a decision tree classifier. By convention, clf means 'classifier'
clf = RandomForestClassifier(n_jobs=-1)

#rank all features, i.e continue the elimination until the last one
rfe = RFE(clf, n_features_to_select=1)
rfe.fit(X_newDoS, Y_DoS)
print ("DoS Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), newcolname_DoS)))




rfe.fit(X_newProbe, Y_Probe)
print ("Probe Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), newcolname_Probe)))




rfe.fit(X_newR2L, Y_R2L)
 
print ("R2L Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), newcolname_R2L)))



rfe.fit(X_newU2R, Y_U2R)
 
print ("U2R Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), newcolname_U2R)))


# # 2. Recursive Feature Elimination, select 13 features each of 122 (Option 2: get 13 best features from 122 from RFE)


from sklearn.feature_selection import RFE
clf = RandomForestClassifier(n_jobs=-1)
rfe = RFE(estimator=clf, n_features_to_select=13, step=1)
rfe.fit(X_DoS, Y_DoS)
X_rfeDoS=rfe.transform(X_DoS)
true=rfe.support_
rfecolindex_DoS=[i for i, x in enumerate(true) if x]
rfecolname_DoS=list(colNames[i] for i in rfecolindex_DoS)
print(true)




rfe.fit(X_Probe, Y_Probe)
X_rfeProbe=rfe.transform(X_Probe)
true=rfe.support_
print("true is ",true)
rfecolindex_Probe=[i for i, x in enumerate(true) if x]
rfecolname_Probe=list(colNames[i] for i in rfecolindex_Probe)




rfe.fit(X_R2L, Y_R2L)
X_rfeR2L=rfe.transform(X_R2L)
true=rfe.support_
rfecolindex_R2L=[i for i, x in enumerate(true) if x]
rfecolname_R2L=list(colNames[i] for i in rfecolindex_R2L)




rfe.fit(X_U2R, Y_U2R)
X_rfeU2R=rfe.transform(X_U2R)
true=rfe.support_
rfecolindex_U2R=[i for i, x in enumerate(true) if x]
rfecolname_U2R=list(colNames[i] for i in rfecolindex_U2R)


# # Summary of features selected by RFE



print('Features selected for DoS:',rfecolname_DoS)
print()
print('Features selected for Probe:',rfecolname_Probe)
print()
print('Features selected for R2L:',rfecolname_R2L)
print()
print('Features selected for U2R:',rfecolname_U2R)




print(X_rfeDoS.shape)
print(X_rfeProbe.shape)
print(X_rfeR2L.shape)
print(X_rfeU2R.shape)
"""

# # Step 4: Build the model:
# ### Classifier is trained for all features and for reduced features, for later comparison.
# #### The classifier model itself is stored in the clf variable.
# all features
# selected features
"""
def create_network():
    model = Sequential()
    model.add(Dense(122,activation='relu',input_shape=(122,),kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(366,activation='relu',kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(60,activation='relu',kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    print(model.summary())
    return model

"""
"""
classifier = KerasClassifier(build_fn=create_network, epochs=10, batch_size=10, verbose=1)
accuracies=cross_val_score(estimator=classifier, X=X_DoS, y=Y_DoS, cv=2)
print("Mean Cross Validated accuracy",np.mean(accuracies))
 """   

# # Step 5: Prediction & Evaluation (validation):
 # Make Prediction for a Single vector -- From the daataset 



def testing(i,model_dos,model_probe,model_r2l,model_u2r):
    #global test_instance
    test_instance=df_test.iloc[i,:].values
    print("----Making Prediction for Input--------")
    print(test_instance)
    test_instance=pd.Series(test_instance,index=col_names)
    final_df=df_test.append(test_instance,ignore_index=True)
    
    categorical_columns=['protocol_type', 'service', 'flag'] 
     # Get the categorical values into a 2D numpy array
   
    testdf_categorical_values =final_df[categorical_columns]
    testdf_categorical_values.head()
    
    
    
    unique_service_test=sorted(final_df.service.unique())
    unique_service2_test=[string2 + x for x in unique_service_test]
    testdumcols=unique_protocol2 + unique_service2_test + unique_flag2
    
    testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)
    
    enc = OneHotEncoder()
    
    # test set
    testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
    testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdumcols)
    
    for col in difference:
        testdf_cat_data[col] = 0

    testdf_cat_data.shape
    
    newdf_test_instance=final_df.join(testdf_cat_data)
    newdf_test_instance.drop('flag', axis=1, inplace=True)
    newdf_test_instance.drop('protocol_type', axis=1, inplace=True)
    newdf_test_instance.drop('service', axis=1, inplace=True)
    
    newdf_test_instance['label'] = newlabeldf_test
    
    newdf_test_instance=newdf_test_instance.drop('label',1)

    
    
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(newdf_test_instance)
    newdf_test_instance=scaler.transform(newdf_test_instance) 
    
    
    test_vec=newdf_test_instance[-1,:].reshape(1,122)
    y_pred_dos=model_dos.predict(test_vec)
    y_pred_probe=model_probe.predict(test_vec)
    y_pred_u2r=model_u2r.predict(test_vec)
    y_pred_r2l=model_r2l.predict(test_vec)
    
    print("|  ----DOS ---- | --- Probe ---- | ---- R2L ---- | ----U2R ---- | ")
    print(y_pred_dos,y_pred_probe,y_pred_r2l,y_pred_u2r)
    
    
    return newdf_test_instance

#working - 5656(mscan --Probe and U2R ) ,6020 (processtable-- for DOS)  789(smurf- DoS , R2L)
#Evaluate using the loaded model
    

json_file = open('model_dos.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_dos = model_from_json(loaded_model_json)
# load weights into new model
model_dos.load_weights("model_dos.h5")
print("Loaded model from disk")


json_file = open('model_probe.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_probe = model_from_json(loaded_model_json)
# load weights into new model
model_probe.load_weights("model_probe.h5")
print("Loaded model from disk")

json_file = open('model_u2r.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_u2r = model_from_json(loaded_model_json)
# load weights into new model
model_u2r.load_weights("model_u2r.h5")
print("Loaded model from disk")

json_file = open('model_r2l.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_r2l = model_from_json(loaded_model_json)
# load weights into new model
model_r2l.load_weights("model_r2l.h5")
print("Loaded model from disk")


#print(df.columns)

#(Normal 5) , rootkit - 9128 sqlattack -> 8125 
#Working Proper 153 (Mailbomb --> DOS)  292 (mscan -> Probe , U2R)  925 --> Buffer OVerflow (U2R)
newdf_test_instance=testing(153,model_dos,model_probe,model_r2l,model_u2r)






"""

model_dos=create_network()
model_dos.fit(X_DoS,Y_DoS,epochs=2,batch_size=10)
score_model_dos = model_dos.evaluate(X_DoS_test, Y_DoS_test,batch_size=10)
print("------Evaluating Accuracy on DOS--------")
print("%s: %.2f%%" % (model_dos.metrics_names[1], score_model_dos[1]*100))
model_dos_json = model_dos.to_json()
with open("model_dos.json", "w") as json_file:
    json_file.write(model_dos_json)
# serialize weights to HDF5
model_dos.save_weights("model_dos.h5")
print("Saved model to disk")


model_r2l=create_network()
model_r2l.fit(X_DoS,Y_DoS,epochs=2,batch_size=10)
score_model_r2l = model_r2l.evaluate(X_DoS_test, Y_DoS_test,batch_size=10)
print("------Evaluating Accuracy on R2L--------")
print("%s: %.2f%%" % (model_r2l.metrics_names[1], score_model_r2l[1]*100))
model_r2l_json = model_r2l.to_json()
with open("model_r2l.json", "w") as json_file:
    json_file.write(model_r2l_json)
# serialize weights to HDF5
model_r2l.save_weights("model_r2l.h5")
print("Saved model to disk")



model_probe=create_network()
model_probe.fit(X_DoS,Y_DoS,epochs=2,batch_size=10)
score_model_probe = model_probe.evaluate(X_DoS_test, Y_DoS_test,batch_size=10)
print("------Evaluating Accuracy on Probe--------")
print("%s: %.2f%%" % (model_probe.metrics_names[1], score_model_probe[1]*100))
model_probe_json = model_probe.to_json()
with open("model_probe.json", "w") as json_file:
    json_file.write(model_probe_json)
# serialize weights to HDF5
model_probe.save_weights("model_probe.h5")
print("Saved model to disk")


model_u2r=create_network()
model_u2r.fit(X_DoS,Y_DoS,epochs=2,batch_size=10)
score_model_u2r = model_u2r.evaluate(X_DoS_test, Y_DoS_test,batch_size=10)
print("------Evaluating Accuracy on U2R--------")
print("%s: %.2f%%" % (model_u2r.metrics_names[1], score_model_u2r[1]*100))
model_u2r_json = model_u2r.to_json()
with open("model_u2r.json", "w") as json_file:
    json_file.write(model_u2r_json)
# serialize weights to HDF5
model_u2r.save_weights("model_u2r.h5")
print("Saved model to disk")

"""

"""
#Evaluate Predictions on Subset of Features
def create_network_reduced():
    model = Sequential()
    model.add(Dense(13,activation='relu',input_shape=(13,),kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(26,activation='relu',kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(5,activation='relu',kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    print(model.summary())
    return model


model_dos_small=create_network_reduced()
model_dos_small.fit(X_rfeDoS,Y_DoS,epochs=2,batch_size=10)
score_model_dos_small = model_dos_small.evaluate(X_rfeDoS, Y_DoS,batch_size=10)
print("----Evaluating DOS Accuracy on Small Features-----")
print("%s: %.2f%%" % (model_dos_small.metrics_names[1], score_model_dos_small[1]*100))

model_probe_small=create_network_reduced()
model_probe_small.fit(X_rfeProbe,Y_Probe,epochs=2,batch_size=10)
score_model_probe_small = model_probe_small.evaluate(X_rfeProbe, Y_Probe,batch_size=10)
print("----Evaluating Probe Accuracy on Small Features-----")
print("%s: %.2f%%" % (model_probe_small.metrics_names[1], score_model_probe_small[1]*100))

model_r2l_small=create_network_reduced()
model_r2l_small.fit(X_rfeR2L,Y_R2L,epochs=2,batch_size=10)
score_model_r2l_small = model_r2l_small.evaluate(X_rfeR2L, Y_R2L,batch_size=10)
print("----Evaluating R2L Accuracy on Small Features-----")
print("%s: %.2f%%" % (model_r2l_small.metrics_names[1], score_model_r2l_small[1]*100))


model_u2r_small=create_network_reduced()
model_u2r_small.fit(X_rfeU2R,Y_U2R,epochs=2,batch_size=10)
score_model_u2r_small = model_u2r_small.evaluate(X_rfeU2R, Y_U2R,batch_size=10)
print("----Evaluating U2R Accuracy on Small Features-----")
print("%s: %.2f%%" % (model_u2r_small.metrics_names[1], score_model_u2r_small[1]*100))

"""

