# Code draws heavily from: https://github.com/JYLFamily/Home_Credit_Default_Risk/blob/master/20180603/FeaturesV2/ApplicationTestFeatures.py

import numpy as np
import pandas as pd 
import re

# Featuretools for automated feature engineering
import featuretools as ft

# Clearing the command console - makes it easier to read
import os
clear = lambda: os.system('cls')
clear()

Project_loc = "C:\\Users\\Cemlyn\\Desktop\\Kaggle\\Home Credit Default Risk\\all\\"

class Feature_Generator(object):

        #--- Defining class constructor ---#
        def __init__(self):

                # Datasets
                self.__application_train = None
                self.__application_test = None
                self.__app =  None
                self.__bureau = None
                self.__bureau_balance = None
                self.__installments_payments = None
                self.__POS_CASH_balance = None
                self.__credit_card_balance = None

                # Dataset Variable_types dictionaries
                self.__app_dict = {}
                self.__previous_app_dict = {}
                self.__bureau_dict = {}
                self.__bureau_balance_dict = {}
                self.__installments_payments_dict = {}
                self.__POS_CASH_balance_dict = {}
                self.__credit_card_balance_dict = {}

                self.__feature_matrix = None
                self.__feature_defs = None


                # es set
                self.__es = None
                self.__feature_dataframe = None

                return None

        ###################################################################################################
        # Declaring variable_types
        ###################################################################################################

        def prepare_data(self):

                print("Defining Meta data...")

                #--------------------------------- Application Data ---------------------------------#
                self.__application_train = pd.read_csv("application_train.csv")
                self.__application_test = pd.read_csv("application_test.csv")
                self.__application_test['TARGET'], self.__application_test['Source'] = np.nan, 'Test'
                self.__application_train['Source'] = 'Train'
                self.__app = pd.concat([self.__application_train, self.__application_test], axis=0)

                #self.__app = self.__app[:100]

                #--- Creating this timestamp to generate relative time intervals between app and previous app
                self.__app['DATE'] = pd.Timestamp("2018-01-01")

                #--- Replacing extreme number with nan ---#
                self.__app[[i for i in self.__app.columns if re.match(r"^DAYS", i)]] = (
                        self.__app[[i for i in self.__app.columns if re.match(r"^DAYS", i)]].replace(365243,np.nan))

                for col in self.__app.columns:
                        if re.match(r"^DAYS", col):
                                self.__app["Days_%s" %(col)] = pd.to_timedelta(self.__app[col],'D')
                                self.__app["DATE_%s" %(col)] = pd.Timestamp("2018-01-01") + self.__app["Days_%s" %(col)]
                                self.__app.drop("Days_%s" %(col),axis=1,inplace=True)

                #--- Replacing these with missing ---#
                self.__app[self.__app.select_dtypes("object").columns.tolist()] = (
                        self.__app[self.__app.select_dtypes("object").columns.tolist()].replace(["XNA", "XAP"],np.nan))

                #--- Creating a dictionary list variables types in app dataset ---#
                for col in self.__app:
                        if self.__app[col].dtype == 'object':
                                self.__app_dict[col] = ft.variable_types.Categorical
                        elif (len(self.__app[col].unique()) <= 2):
                                self.__app_dict[col] = ft.variable_types.Boolean
                        elif self.__app[col].dtype == 'datetime64[ns]':
                                self.__app_dict[col] = ft.variable_types.Datetime                      
                        else:
                                self.__app_dict[col] = ft.variable_types.Numeric

                self.__app_dict['REGION_RATING_CLIENT'] = ft.variable_types.Ordinal 
                self.__app_dict['REGION_RATING_CLIENT_W_CITY'] = ft.variable_types.Ordinal


                #--------------------------------- Previous Application Data ---------------------------------#
                self.__previous_application = pd.read_csv("previous_application.csv")

                #--- Replacing extreme number with nan ---#
                self.__previous_application[[i for i in self.__previous_application.columns if re.match(r"^DAYS", i)]] = (
                        self.__previous_application[[i for i in self.__previous_application.columns if re.match(r"^DAYS", i)]].replace(365243,np.nan))

                for col in self.__previous_application.columns:
                        if re.match(r"^DAYS", col):
                                self.__previous_application["Days_%s" %(col)] = pd.to_timedelta(self.__previous_application[col],'D')
                                self.__previous_application["DATE_%s" %(col)] = pd.Timestamp("2018-01-01") + self.__previous_application["Days_%s" %(col)]
                                self.__previous_application.drop("Days_%s" %(col),axis=1,inplace=True)

                self.__previous_application[self.__previous_application.select_dtypes("object").columns.tolist()] = (
                        self.__previous_application[self.__previous_application.select_dtypes("object").columns.tolist()].replace(["XNA", "XAP"],np.nan))

                for col in self.__previous_application.columns:
                        if self.__previous_application[col].dtype == 'object':
                                self.__previous_app_dict[col] = ft.variable_types.Categorical
                        elif (len(self.__previous_application[col].unique()) <= 2):
                                self.__previous_app_dict[col] = ft.variable_types.Boolean
                        elif self.__previous_application[col].dtype == 'datetime64[ns]':
                                self.__previous_app_dict[col] = ft.variable_types.Datetime
                        else:
                                self.__previous_app_dict[col] = ft.variable_types.Numeric

                self.__previous_app_dict['SELLERPLACE_AREA'] = ft.variable_types.Categorical


                #--------------------------------- Bureau Data ---------------------------------#
                self.__bureau = pd.read_csv("bureau.csv")

                #--- Replacing extreme number with nan ---#
                self.__bureau[[i for i in self.__bureau.columns if re.match(r"^DAYS", i)]] = (
                        self.__bureau[[i for i in self.__bureau.columns if re.match(r"^DAYS", i)]].replace(365243,np.nan))

                for col in self.__bureau.columns:
                        if re.match(r"^DAYS", col):
                                self.__bureau["Days_%s" %(col)] = pd.to_timedelta(self.__bureau[col],'D')
                                self.__bureau["DATE_%s" %(col)] = pd.Timestamp("2018-01-01") + self.__bureau["Days_%s" %(col)]
                                self.__bureau.drop("Days_%s" %(col),axis=1,inplace=True)

                self.__bureau[self.__bureau.select_dtypes("object").columns.tolist()] = (
                        self.__bureau[self.__bureau.select_dtypes("object").columns.tolist()].replace(["XNA", "XAP"],np.nan))

                for col in self.__bureau.columns:
                        if self.__bureau[col].dtype == 'object':
                                self.__bureau_dict[col] = ft.variable_types.Categorical
                        elif (len(self.__bureau[col].unique()) <= 2):
                                self.__bureau_dict[col] = ft.variable_types.Boolean
                        elif self.__bureau[col].dtype == 'datetime64[ns]':
                                self.__bureau_dict[col] = ft.variable_types.Datetime
                        else:
                                self.__bureau_dict[col] = ft.variable_types.Numeric

                #--------------------------------- Bureau_balance Data ---------------------------------#
                self.__bureau_balance = pd.read_csv("bureau_balance.csv")

                self.__bureau_balance['MONTHS_BALANCE'] =  pd.to_timedelta(self.__bureau_balance['MONTHS_BALANCE'],'M')
                self.__bureau_balance['DATE_MONTHS_BALANCE'] =  pd.Timestamp("2018-01-01") + self.__bureau_balance['MONTHS_BALANCE']

                self.__bureau_balance[self.__bureau_balance.select_dtypes("object").columns.tolist()] = (
                        self.__bureau_balance[self.__bureau_balance.select_dtypes("object").columns.tolist()].replace(["XNA", "XAP"],np.nan))

                for col in self.__bureau_balance.columns:
                        if self.__bureau_balance[col].dtype == 'object':
                                self.__bureau_balance_dict[col] = ft.variable_types.Categorical
                        elif (len(self.__bureau_balance[col].unique()) <= 2):
                                self.__bureau_balance_dict[col] = ft.variable_types.Boolean
                        elif self.__bureau_balance[col].dtype == 'datetime64[ns]':
                                self.__bureau_balance_dict[col] = ft.variable_types.Datetime
                        else:
                                self.__bureau_balance_dict[col] = ft.variable_types.Numeric


                #--------------------------------- POS_CASH_balance Data ---------------------------------#
                self.__POS_CASH_balance = pd.read_csv("POS_CASH_balance.csv")

                self.__POS_CASH_balance[self.__POS_CASH_balance.select_dtypes("object").columns.tolist()] = (
                        self.__POS_CASH_balance[self.__POS_CASH_balance.select_dtypes("object").columns.tolist()].replace(["XNA", "XAP"],np.nan))

                self.__POS_CASH_balance['MONTHS_BALANCE'] =  pd.to_timedelta(self.__POS_CASH_balance['MONTHS_BALANCE'],'M')
                self.__POS_CASH_balance['DATE_MONTHS_BALANCE'] =  pd.Timestamp("2018-01-01") + self.__POS_CASH_balance['MONTHS_BALANCE']

                for col in self.__POS_CASH_balance.columns:
                        if self.__POS_CASH_balance[col].dtype == 'object':
                                self.__POS_CASH_balance_dict[col] = ft.variable_types.Categorical
                        elif (len(self.__POS_CASH_balance[col].unique()) <= 2):
                                self.__POS_CASH_balance_dict[col] = ft.variable_types.Boolean
                        elif self.__POS_CASH_balance[col].dtype == 'datetime64[ns]':
                                self.__POS_CASH_balance_dict[col] = ft.variable_types.Datetime
                        else:
                                self.__POS_CASH_balance_dict[col] = ft.variable_types.Numeric


                #--------------------------------- Installments_payments.csv Data ---------------------------------#
                self.__installments_payments = pd.read_csv("installments_payments.csv")

                self.__installments_payments[self.__installments_payments.select_dtypes("object").columns.tolist()] = (
                        self.__installments_payments[self.__installments_payments.select_dtypes("object").columns.tolist()].replace(["XNA", "XAP"],np.nan))

                #--- Replacing extreme number with nan ---#
                self.__installments_payments[[i for i in self.__installments_payments.columns if re.match(r"^DAYS", i)]] = (
                        self.__installments_payments[[i for i in self.__installments_payments.columns if re.match(r"^DAYS", i)]].replace(365243,np.nan))

                for col in self.__installments_payments.columns:
                        if re.match(r"^DAYS", col):
                                self.__installments_payments["Days_%s" %(col)] = pd.to_timedelta(self.__installments_payments[col],'D')
                                self.__installments_payments["DATE_%s" %(col)] = pd.Timestamp("2018-01-01") + self.__installments_payments["Days_%s" %(col)]
                                self.__installments_payments.drop("Days_%s" %(col),axis=1,inplace=True)


                for col in self.__installments_payments.columns:
                        if self.__installments_payments[col].dtype == 'object':
                                self.__installments_payments_dict[col] = ft.variable_types.Categorical
                        elif (len(self.__installments_payments[col].unique()) <= 2):
                                self.__installments_payments_dict[col] = ft.variable_types.Boolean
                        elif self.__installments_payments[col].dtype == 'datetime64[ns]':
                                self.__installments_payments_dict[col] = ft.variable_types.Datetime
                        else:
                                self.__installments_payments_dict[col] = ft.variable_types.Numeric


                #--------------------------------- credit_card_balance.csv Data ---------------------------------#
                self.__credit_card_balance = pd.read_csv("credit_card_balance.csv")

                self.__credit_card_balance[self.__credit_card_balance.select_dtypes("object").columns.tolist()] = (
                        self.__credit_card_balance[self.__credit_card_balance.select_dtypes("object").columns.tolist()].replace(["XNA", "XAP"],np.nan))

                #--- Replacing extreme number with nan ---#
                self.__credit_card_balance[[i for i in self.__credit_card_balance.columns if re.match(r"^DAYS", i)]] = (
                        self.__credit_card_balance[[i for i in self.__credit_card_balance.columns if re.match(r"^DAYS", i)]].replace(365243,np.nan))

                self.__credit_card_balance['MONTHS_BALANCE'] =  pd.to_timedelta(self.__credit_card_balance['MONTHS_BALANCE'],'M')
                self.__credit_card_balance['DATE_MONTHS_BALANCE'] =  pd.Timestamp("2018-01-01") + self.__credit_card_balance['MONTHS_BALANCE']
                
                for col in self.__credit_card_balance.columns:
                        if re.match(r"^DAYS", col):
                                self.__credit_card_balance["Days_%s" %(col)] = pd.to_timedelta(self.__credit_card_balance[col],'D')
                                self.__credit_card_balance["DATE_%s" %(col)] = pd.Timestamp("2018-01-01") + self.__credit_card_balance["Days_%s" %(col)]
                                self.__credit_card_balance.drop("Days_%s" %(col),axis=1,inplace=True)


                for col in self.__credit_card_balance.columns:
                        if self.__credit_card_balance[col].dtype == 'object':
                                self.__credit_card_balance_dict[col] = ft.variable_types.Categorical
                        elif (len(self.__credit_card_balance[col].unique()) <= 2):
                                self.__credit_card_balance_dict[col] = ft.variable_types.Boolean
                        elif self.__credit_card_balance[col].dtype == 'datetime64[ns]':
                                self.__credit_card_balance_dict[col] = ft.variable_types.Datetime
                        else:
                                self.__credit_card_balance_dict[col] = ft.variable_types.Numeric


                #--- Declaring meta data in entity ---#
                self.__es = ft.EntitySet(id="app")

                self.__es = self.__es.entity_from_dataframe(
                        entity_id="app",
                        dataframe=self.__app,
                        index="SK_ID_CURR",
                        variable_types=self.__app_dict)

                self.__es = self.__es.entity_from_dataframe(
                        entity_id="previous_app",
                        dataframe=self.__previous_application,
                        index="SK_ID_PREV",
                        time_index="DATE_DAYS_DECISION",
                        variable_types=self.__previous_app_dict)

                self.__es = self.__es.entity_from_dataframe(
                        entity_id="bureau",
                        dataframe=self.__bureau,
                        index="SK_ID_BUREAU",
                        time_index="DATE_DAYS_CREDIT",
                        variable_types=self.__bureau_dict)
                
                self.__es = self.__es.entity_from_dataframe(
                        entity_id="bureau_balance",
                        dataframe=self.__bureau_balance,
                        make_index = True, index = 'bb_index',
                        time_index="DATE_MONTHS_BALANCE",
                        variable_types=self.__bureau_balance_dict)

                self.__es = self.__es.entity_from_dataframe(
                        entity_id="POS_CASH",
                        dataframe=self.__POS_CASH_balance,
                        make_index = True, index = 'POS_index',
                        time_index="DATE_MONTHS_BALANCE",
                        variable_types=self.__POS_CASH_balance_dict)

                self.__es = self.__es.entity_from_dataframe(
                        entity_id="Install",
                        dataframe=self.__installments_payments,
                        make_index = True, index = 'Inst_index',
                        time_index="DATE_DAYS_INSTALMENT",
                        variable_types=self.__installments_payments_dict)

                self.__es = self.__es.entity_from_dataframe(
                        entity_id="credt_card",
                        dataframe=self.__credit_card_balance,
                        make_index = True, index = 'CC_index',
                        time_index="DATE_MONTHS_BALANCE",
                        variable_types=self.__credit_card_balance_dict)


                #--- Defining Relationships ---#
                prev_app_re = ft.Relationship(self.__es['app']['SK_ID_CURR'], self.__es['previous_app']['SK_ID_CURR'])
                bureau_app_re = ft.Relationship(self.__es['app']['SK_ID_CURR'], self.__es['bureau']['SK_ID_CURR'])
                bureau_bal_re = ft.Relationship(self.__es['bureau']['SK_ID_BUREAU'], self.__es['bureau_balance']['SK_ID_BUREAU'])
                prev_pos_re = ft.Relationship(self.__es['previous_app']['SK_ID_PREV'], self.__es['POS_CASH']['SK_ID_PREV'])
                prev_inst_re = ft.Relationship(self.__es['previous_app']['SK_ID_PREV'], self.__es['Install']['SK_ID_PREV'])
                prev_cc_re = ft.Relationship(self.__es['previous_app']['SK_ID_PREV'], self.__es['credt_card']['SK_ID_PREV'])

                self.__es = self.__es.add_relationship(prev_app_re)
                self.__es = self.__es.add_relationship(bureau_app_re)
                self.__es = self.__es.add_relationship(bureau_bal_re)
                self.__es = self.__es.add_relationship(prev_pos_re)
                self.__es = self.__es.add_relationship(prev_inst_re)
                self.__es = self.__es.add_relationship(prev_cc_re)

        #--- Generating new features ---#
        def es_set(self):

                print("Generating Features...\n")

                # List the primitives in a dataframe
                primitives = ft.list_primitives()
                pd.options.display.max_colwidth = 100
                print("feature primitives:", primitives[primitives['type'] == 'aggregation'].head(10))

                self.__feature_matrix, self.__feature_defs = ft.dfs(entityset=self.__es,target_entity="app", verbose=True)

                return self.__feature_matrix

Gen = Feature_Generator()
Gen.prepare_data()
app = Gen.es_set()

app.to_csv("DFS_feature_matrix.csv")
app.to_pickle("C:\\Users\\Cemlyn\\OneDrive\\Python_Code_Repository\\DFS_feature_matrix.pkl")