#  v00 - First Draft
# v01 - checking for non-linearity in variables and transforming as appropriate

import pandas as pd 
import numpy as np
import re
import itertools
import time

import warnings
#warnings.filterwarnings('error')

# Importing Packages for preprocessing
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, Imputer

# PCA for dimensionality reduction on the categorcial variables
from sklearn.decomposition import PCA, KernelPCA

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV , KFold
from time import time

import lightgbm as lgb

# category endcoder
from category_encoders import TargetEncoder

# Clearing the command console - makes it easier to read
import os
clear = lambda: os.system('cls')
clear()

np.random.seed(0)

start_time = time()
print("Commencing..... \n")

# Reading in Data
df = pd.read_csv("DFS_feature_matrix.csv", index_col='SK_ID_CURR')


def preprocessing(data):

	#--- Drop columns where all values are missing. Do this first to try and save space ---#
	data.dropna(how='all', axis=1, inplace=True)

	#########################################################################################################################
	# Creating Some Additional Variables
	#########################################################################################################################

	print("Generating some domain knowledge features...")

	# Loan_to_income Ratio (LTV)
	data['LOAN_INCOME_RATIO'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']

	# Inc to Anuity Ratio
	data['ANNUITY_INCOME_RATIO'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']

	# Income to Collateral Ratio
	data['COLLATERAL_INCOME_RATIO'] =  data['AMT_GOODS_PRICE'] / data['AMT_INCOME_TOTAL']

	# LTV
	data['LOAN_TO_VALUE_RATIO'] = data['AMT_CREDIT'] / data['AMT_GOODS_PRICE']

	# Stats on the external scores
	data['EXT_SOURCE_MEAN'] = data[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
	data['EXT_SOURCE_MIN'] = data[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].min(axis=1)
	data['EXT_SOURCE_MAX'] = data[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].max(axis=1)
	data['EXT_SOURCE_STD'] = data[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].std(axis=1)
	data['EXT_SOURCE_SKEW'] = data[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].skew(axis=1)

	#Income to no. kids ratio
	data['INC_TO_KIDS'] = data['AMT_INCOME_TOTAL'] / (data['CNT_CHILDREN'] + 1)

	#Fraction of family kids
	data['PERC_KIDS'] = data['CNT_CHILDREN'] / data['CNT_FAM_MEMBERS']

	#indebtendess*kids
	data['KIDS_AMT_ANNUITY_PRODUCT'] = data['CNT_CHILDREN']*data['AMT_ANNUITY']

	#Fraction of life worked
	data['WORK_FRAC'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']

	#Days old wehn got first car
	data['FIRST_CAR_DAYS'] = -data['DAYS_BIRTH'] - 365*data['OWN_CAR_AGE']

	# Total ways in which customer can be contacted
	data['SUM_CONTACT'] = data['FLAG_MOBIL'] + data['FLAG_EMP_PHONE'] + data['FLAG_WORK_PHONE'] + data['FLAG_CONT_MOBILE'] + data['FLAG_PHONE'] + data['FLAG_EMAIL']

	# Age income product
	data['AGE_INCOME_PROD'] = data['DAYS_BIRTH']*data['AMT_INCOME_TOTAL']

	# Working Age income product
	data['EMPLOYED_INCOME_PROD'] = data['DAYS_EMPLOYED']*data['AMT_INCOME_TOTAL']

	#########################################################################################################################
	# Dealing with categorical data (columns with string or object values) - filling in missings and creating dummy variables
	#########################################################################################################################

	print("filling in missing categorical data...")

	cat_data = data.select_dtypes(['object'])
	cat_col = list(data.select_dtypes(['object']).columns.values)
	cat_col.remove('Source')


	#--- Creating a list of categorical variabels with missing rows and filling in with string 'missing_' ---#
	cat_miss_col = cat_data.columns[cat_data.isna().any()].tolist()
	for item in cat_miss_col:
		data['%s' %(item)].fillna('missing_', inplace=True)

	encoder = TargetEncoder(verbose=0, impute_missing=True, return_df=False, smoothing=1)

	encoder.fit(X=data.loc[data['Source']=='Train',cat_col].values,
		y=data.loc[data['Source']=='Train',['TARGET']].values.reshape(-1,))

	X = encoder.transform(X=data[cat_col].values)

	data.loc[:,cat_col] = pd.DataFrame(X, columns=cat_col, index=list(data.index))


	################################################################################################################################
	# Dealing with floating values - imputing and then normalising
	################################################################################################################################

	print("Imputing, normalisig and scaling...")

	# Initialising preprocessing to normalise, scale data and impute missing data
	normaliser = Normalizer()
	scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
	imputer = Imputer(axis=0, strategy="median", missing_values="NaN")

	# Method to impute, scale and transform numerical columns into gaussian distribution
	column_list = list(data.columns)
	column_list.remove('TARGET')
	column_list.remove('Source')

	print("Imputing....")
	imputer.fit(data.loc[data['Source']=='Train',column_list])
	data.loc[:,column_list] = imputer.transform(data[column_list])

	print("Scaling....")
	scaler.fit(data.loc[data['Source']=='Train',column_list])
	data.loc[:,column_list] = scaler.transform(data[column_list])

	print("Normalising....")
	normaliser.fit(data.loc[data['Source']=='Train',column_list])
	data.loc[:,column_list] = normaliser.transform(data[column_list])


	#--- Deleting all zero columns ---#
	data = data.loc[:, (data != 0).any(axis=0)]

	#--- Getting list of variables in the dataset and correlations ---#
	
	correlations = []
	index = []

	for col in list(data.columns):
		if col!='TARGET' and col!='Source':
			correlations.append(round(data['TARGET'].corr(data[col]),3))
			index.append(col)

	correlations = pd.DataFrame(data=correlations, index=index, columns=['Correlation'])
	correlations['abs_Correlation'] = abs(correlations['Correlation'])
	correlations.to_csv("correlations.csv")

	################################################################################################################################
	# For highly correlated variables create a news variables
	################################################################################################################################

	# Creating products and divisor features for all numerical variables
	correlations = correlations[correlations['abs_Correlation']>0.05]
	num_vars = list(correlations.index.values)

	combinations = itertools.combinations(num_vars,2)
	
	print("Generating %s new features...." %(3*len(list(combinations))))

	start_time_2 = time()

	iteration = 0

	for i, j in itertools.combinations(num_vars,2):

		iteration+=1

		data['PROD_%s__%s' %(i,j)] = data[i]*data[j]
		data['DIV_%s__%s' %(i,j)] = data[i]/data[j]
		data['DIV2_%s__%s' %(i,j)] = data[j]/data[i]

		print("iteration: %s, time:" %(iteration,time()-start_time_2))

	correlations = []
	index = []

	for col in list(data.columns):
		if col!='TARGET' and col!='Source':
			correlations.append(round(data['TARGET'].corr(data[col]),3))
			index.append(col)

	correlations = pd.DataFrame(data=correlations, index=index, columns=['Correlation'])
	correlations['abs_Correlation'] = abs(correlations['Correlation'])
	correlations.to_csv("correlations_2.csv")
	correlations.to_csv("C:\\Users\\Cemlyn\\OneDrive\\Python_Code_Repository\\correlations_2.csv")
	'''
	print("Creating non-linear versions of numeric variables...")

	#--- If variable if float or number and non-binary then check for non-linear relationships ---#
	dtypes = list(set(data.dtypes))'''

	#--- Convert all int numbers to float - this will be memory intensive ---#
	'''
	power_list = [1.0,2.0,3.0]

	for col in data:

		#--- if column is numeric and non-binary then create new versions
		if data[col].dtype!='object' and len(data[col].unique()) > 2:

			corr_list={}

			data["%s_sqrt" %(col)] = np.sqrt(np.abs(data[col]))
			corr_list['sqrt'] = data['TARGET'].corr(data["%s_sqrt" %(col)])

			for power in power_list:
				data["%s_%s" %(col,power)] = np.power(data[col],power)

				corr = data['TARGET'].corr(data["%s_%s" %(col,power)])

				corr_list[power] = corr

			#--- if non-linearised variable has higher correlation keep the non-linear form which has the highest correlation
			if data['TARGET'].corr(data[col])<max(corr_list.values()):

				data.drop("%s" %(col),axis=1,inplace=True)

				for x in corr_list.keys():
					if x!=max(corr_list.values()):
						

						if ("%s_%s" %(col,power)) in data.columns:
							data.drop("%s_%s" %(col,power),axis=1,inplace=True)
		
	print(data.info())
	'''
	# Converting all int64 to int32 to save space. Might do the same with float64
	'''
	dtypes = list(set(data.dtypes))
	for types in dtypes:
		df_type = list(data.select_dtypes(types).columns)

		if 'TARGET' in df_type:
			df_type.remove('TARGET')

		#Convert 64bit values to 32 to save space
		if types=='int64':
			for col in df_type:
				data[col] = data[col].astype('int32')
		if types=='float64':
			for col in df_type:
				data[col] = data[col].astype('float32')

		df_type = data.select_dtypes(types).columns
	'''

	print(data.info())

	data.to_pickle("Processed_DFS_Data_v03.pkl")
	data.to_pickle("C:\\Users\\Cemlyn\\OneDrive\\Python_Code_Repository\\Processed_DFS_Data_v03.pkl")
	#data[:1000].to_csv("Processed_DFS_Data_sampled.csv")

	return 0

preprocessing(df)


print("--- %s seconds ---" % (time() - start_time))