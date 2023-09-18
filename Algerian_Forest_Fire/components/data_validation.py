#Comparing the columns and changing its data types and finally checking the data drift and saving the drift report

from Algerian_Forest_Fire.configuration.Algerian_forest_config import *
from Algerian_Forest_Fire.constants.constants import *
from Algerian_Forest_Fire.entity.entity import *
from Algerian_Forest_Fire.exception import ForestFireException
from Algerian_Forest_Fire.logger import logging
from Algerian_Forest_Fire.configuration.Algerian_forest_config import *
import os,sys
import urllib.request
from datetime import datetime
import tarfile
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from pandas_profiling import ProfileReport
import zipfile
from utility.util import *

validation_config = data_validation_config()

class data_validation:
    def __init__(self, x_train , x_test , csv_file):
        self.csv_file = csv_file
        self.x_train = x_train
        self.x_test = x_test

    def comparing_features(self):

        # importing the data_ingestion_configuration    
        # comparing the columns present in the ingested data


        ingested_data = pd.read_csv(self.csv_file, skiprows=1) 
        #comparing columns of both of the data sets
        flag = 1
        print(list(ingested_data.columns) )
        print(validation_config.get_columns_name())
         

        if (set(ingested_data.columns) == set(validation_config.get_columns_name())):
            flag = 1
            logging.info('Schema validated Successfully {No Change in column features}')
            print('matched')

        else:
            logging.info("Column features don't match with schema")
            flag =0
            print('not matched')


        # converting data types of the features that will 
        
        if (flag == 1):
            columns_schema_file = [ column_name for column_name in validation_config.get_float_data_types_columns() if column_name!= 'Temperature']
            columns_in_Ingested_data = [ i for i in ingested_data.columns if i not in ['Classes  ', 'Temperature']]
            if set(columns_in_Ingested_data) == set(columns_schema_file):
                
                print('Columns are validated')
            else:
                print('Columns are not validated')
                print(columns_in_Ingested_data)
                print(columns_schema_file)


    def checking_data_drift(self):

                Validation_status = None
                logging.info('Successfully changed the Dtypes Checking data drift')
                report = Report(metrics=[DataDriftPreset(), ])
                X_train = pd.read_csv(self.x_train)
                X_test = pd.read_csv(self.x_test)
                print(X_train)
                print(X_test)

                report.run(reference_data=X_train, current_data=X_test)
                report.save_html('Data_Drift_Report.html')
                
                metrics_list = report.as_dict()['metrics']
                df_rows = []

                for metric_data in metrics_list:
                    metric = metric_data['metric']
                    result = metric_data['result']
                    df_rows.append({'metric': metric, **result})

                df = pd.DataFrame(df_rows)
                for i in df[['dataset_drift']].loc[1]:
                    answer = i
                
                if answer==True:
                    print('Data Drift Detected')
                    logging.info('Data Drift Detected')
                    Validation_status = False
                else:
                    print('No Data Drift Detected good to go')
                    logging.info('No Data Drift Detected good to go')
                    Validation_status=True

                source_path = os.path.abspath('Data_Drift_Report.html')

                destination_evidently_report_loc = os.path.join(time_stamp_dir_loc, "Data_Drift_report")
                os.makedirs(destination_evidently_report_loc, exist_ok = True)
                shutil.move(source_path,destination_evidently_report_loc)
                return Validation_status

    def generate_profile_report_raw_data(self):
            X_train = pd.read_csv(self.x_train)

            profile_report_location_for_html = os.path.join(root_dir,'Templates','PandasProfiling_before')
            os.makedirs(profile_report_location_for_html , exist_ok=True) #imp changes required

            
            destination_profile_report_dir_loc = os.path.join(time_stamp_dir_loc , "Data_Analysis_report")
            os.makedirs(destination_profile_report_dir_loc, exist_ok = True)

            # Initiating profile report 

            profile_report_location = os.path.join(destination_profile_report_dir_loc, 'Pandas_profiling.html')

            profile = ProfileReport(X_train)

            
            # Generate the report and save it
            profile.to_file(profile_report_location)

            #moving this file to the new location inside template for html access

            shutil.copy(profile_report_location,profile_report_location_for_html)
            return profile_report_location

    def initiate_data_validation(self):

        self.comparing_features()
        status = self.checking_data_drift()
        report_loc = self.generate_profile_report_raw_data()
        validation_output = data_validation_entity(status , report_loc)
        return validation_output



  

trainLoc =  'C:\\\\Users\\\\Lokesh\\\\Desktop\\\\Ineurone\\\\Project\\\\Algerian Forest Fire Full\\\\artifact\\\\2023-16-09_17-35-04\\\\reg_train_test_data\\\\reg_train_data\\\\train_input.csv' 
testloc = 'C:\\\\Users\\\\Lokesh\\\\Desktop\\\\Ineurone\\\\Project\\\\Algerian Forest Fire Full\\\\artifact\\\\2023-16-09_17-35-04\\\\reg_train_test_data\\\\reg_test_data\\\\test_input.csv' 

complete_csv='C:\\\\Users\\\\Lokesh\\\\Desktop\\\\Ineurone\\\\Project\\\\Algerian Forest Fire Full\\\\artifact\\\\2023-16-09_17-35-04\\\\csv_file\\\\Algerian_forest_fires_dataset_UPDATE.csv'
obj = data_validation(trainLoc,testloc,complete_csv)
print(obj.initiate_data_validation()
)       

  