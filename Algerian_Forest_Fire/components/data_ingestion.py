 #creating artifact directory
    # Creating a new directory based on the time stamp
     # creating a new directory to save extracted data
     # Creating a file path of the name of the file you are extracting
       # Downloading the file to the location of the name of the file you have created on the above step.
       # creating a new folder to save the data of the tar file ( The file you have downloaded in the above step)
       # extracting this file to the given file location ( location name : csv_file_location )
       # Using pandas to read and split the data into train and test 
       # creating a seprate folder to store training and testing files
       # Moving the training and testing data to the following location ( you also need to create sperate training and target files inside)
       # First converted the file location url to raw string and then moved to specific location since pandas does not accept any other url
       # Save the location of train x , train y , test x , test y to our named tuple ENTITY of data_ingestion

from Algerian_Forest_Fire.configuration.Algerian_forest_config import data_ingestion_config
from Algerian_Forest_Fire.constants.constants import *
from Algerian_Forest_Fire.entity.entity import data_ingestion_entity
from Algerian_Forest_Fire.exception import ForestFireException
from Algerian_Forest_Fire.logger import logging
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np
import os,sys
import urllib.request
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import shutil

import zipfile
from utility.util import *

 

 
ingestion_config = data_ingestion_config()
download_url = ingestion_config.get_url()
class data_ingestion_component:


    def __init__(self , url=download_url):
        self.url = url

    def creating_dir(self):
                # Creating artifact dir 
        try:

            os.makedirs(artifact_dir , exist_ok=True)

            # creating a new dirctory based on time stamp

            os.makedirs(time_stamp_dir_loc , exist_ok=True)


            # creating a new dirctory to save the extracted data ( file name : raw_data)

            os.makedirs(ingestion_config.get_raw_data_dir_loc(), exist_ok = True)

            # creating a new file location which also contains the name of your file 

 
            logging.info('Data Ingestion Directory successfully created')

        except Exception as e:
            raise ForestFireException(e, sys) from e


    def downloading_files(self):
        try:
            logging.info('Downloading the files')

            download_url = self.url
            urllib.request.urlretrieve(download_url, ingestion_config.get_tgz_file_loc() )

            logging.info('file downloaded successfully')


        # creating a new folder to save the data of the tar file 

            os.makedirs(ingestion_config.get_csv_file_location() , exist_ok=True)
    

        # extracting this file to the given file location ( csv_file_location )

        

            logging.info('Extracting the Zip file')



            with zipfile.ZipFile( ingestion_config.get_tgz_file_loc()   , 'r') as zip_f:
                   zip_f.extractall(ingestion_config.get_csv_file_location())


            logging.info('Zip file extracted')

        except Exception as e:
            raise ForestFireException(e, sys) from e

  

    def csv_conversion(self):
        try:
             # downloading a file to the above location created for regression as well as classification


             os.makedirs(ingestion_config.get_file_location_reg() , exist_ok=True)

 
            # using r join since pandas only takes raw string 
 
             raw_dynamic_path_reg = r"\\".join( ingestion_config.get_final_csv_location().split("\\"))

         #imp   raw_dynamic_path_classi = r"\\".join(file_location_classi.split("\\"))



            # ( for removing headings we can use skip rows)
    

             
              
             return raw_dynamic_path_reg
           #imp data_classi = pd.read_csv(raw_dynamic_path_classi, skiprows=1)

        
        
  
    

        except Exception as e:
            raise ForestFireException(e, sys) from e


    def data_cleaning(self):

              # Cleaning the data :

        

        data_final_loc = self.csv_conversion()
        data = pd.read_csv(data_final_loc, skiprows=1)        
        print(data.head())


        # Filetering dataframe 

        index_to_remove = []

        index_to_remove.append(data[data['Classes  '] == 'Classes  '].index)
        index_to_remove.append(data[data['day'] == 'Sidi-Bel Abbes Region Dataset'].index)
        index_to_remove.append(data[data['DC'] == '14.6 9'].index)
        index_to_remove.append(data[data['FWI'] == 'fire   '].index)
        for i in index_to_remove:
            try:
                data.drop(index  = i, inplace = True)
            except Exception as e:
                print(e)

 

        replacement_mapping = {
                            'fire   ': 'fire',
                            'fire ': 'fire',
                            'fire': 'fire',
                            'not fire   ': 'not fire',
                            'not fire     ': 'not fire',
                            'not fire': 'not fire',
                            'not fire    ': 'not fire',
                            'not fire ': 'not fire',
                            'Classes  ': None,  # Remove 'Classes  '
                            None: None  # Remove NaN values (if any)
                        }

            # Replace values in the 'Classes' column using the mapping dictionary
        data['Classes  '] = data['Classes  '].map(replacement_mapping)


        # Again mapping the data and converting fire and non fire in one and 0
        replacement_mapping = {
                            'fire': 1,
                             
                            'not fire': 0,
                             
                        }

            # Replace values in the 'Classes' column using the mapping dictionary
        data['Classes  '] = data['Classes  '].map(replacement_mapping)
        data['Classes  '] =data['Classes  '].astype(int)

        # Converting the data types to float for machine learning model input 
         

        for i in [ column_names for column_names in data.columns if column_names != 'Classes  ']:
            print(i)
            data[i]= data[i].astype(float)

        # Handeling Null values
        imputer = SimpleImputer(strategy = 'median')
        transformed_data = imputer.fit_transform(data)
        # Handeling Outliers ########################
    

        z_threshold = ingestion_config.get_z_threshold()
 
                    # Calculate Z-Scores for the columns you're interested in or select whole
        z_scores = np.abs(stats.zscore(transformed_data))

                    # Create a boolean mask to identify outliers

                    # This will return true and fase with all the values which is smaller than 
        outlier_mask = (z_scores > z_threshold).any(axis=1)

                
        df_no_outliers_ = data[~outlier_mask]
        df_no_outliers_ = pd.DataFrame(df_no_outliers_, columns = data.columns)
          
        return df_no_outliers_


    def reg_classi_splitting_data_saving_to_dir(self):

        try:

            data_final = self.data_cleaning()
            data_final = pd.DataFrame(data_final)
             

            input_reg = data_final.drop(columns = 'Temperature')
            target_reg = data_final['Temperature']
            target_reg = pd.DataFrame(target_reg)
            

 
            reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(input_reg, target_reg, test_size=0.10, random_state=42)

            # creating a seprate folder to store training and testing files

            logging.info('Seperating Training and Testing Data')

            os.makedirs(ingestion_config.get_reg_train_test_data_dir(), exist_ok=True)

            os.makedirs(ingestion_config.get_reg_train_data_dir(), exist_ok=True)


            os.makedirs(ingestion_config.get_reg_test_data_dir(), exist_ok=True)

            # Moving the training and testing data to the following location name of file is
            #imp else it will show error.

            reg_X_location_for_train_input = os.path.join(ingestion_config.get_reg_train_data_dir(),'train_input.csv')
            reg_Y_location_for_train_input = os.path.join(ingestion_config.get_reg_train_data_dir(), 'train_target.csv')

            reg_X_test_input_location = os.path.join(ingestion_config.get_reg_test_data_dir(),"test_input.csv")
            reg_Y_test_input_location = os.path.join(ingestion_config.get_reg_test_data_dir(), "test_target.csv")


            # using r join since pandas only takes raw string on train test

            reg_X_location_for_train_input = r"\\".join(reg_X_location_for_train_input.split("\\"))   

            reg_Y_location_for_train_input = r"\\".join(reg_Y_location_for_train_input.split("\\"))



            # using r join since pandas only takes raw string on test test

            reg_X_test_input_location = r"\\".join(reg_X_test_input_location.split("\\"))
            reg_Y_test_input_location = r"\\".join(reg_Y_test_input_location.split("\\"))

            reg_X_train.to_csv( reg_X_location_for_train_input)
            reg_y_train.to_csv(reg_Y_location_for_train_input)

            reg_X_test.to_csv( reg_X_test_input_location)
            reg_y_test.to_csv(reg_Y_test_input_location)
            logging.info('Training and testing data seperated successfully for Regression')



########################################### For Classification #############################################################

            
            

            input_classi = data_final.drop(columns = ['Classes  '])
            target_classi = data_final['Classes  ']
            target_classi = pd.DataFrame(target_classi)
 
            classi_X_train, classi_X_test, classi_y_train, classi_y_test = train_test_split(input_classi, target_classi, test_size=0.10, random_state=42)

            # creating a seprate folder to store training and testing files

            logging.info('Seperating Training and Testing Data')

            os.makedirs(ingestion_config.get_classi_train_test_data_dir(), exist_ok=True)

            os.makedirs(ingestion_config.get_classi_train_data_dir(), exist_ok=True)


            os.makedirs(ingestion_config.get_classi_test_data_dir(), exist_ok=True)

            # Moving the training and testing data to the following location name of file is
            #imp else it will show error.

            classi_X_location_for_train_input = os.path.join(ingestion_config.get_classi_train_data_dir(),'train_input.csv')
            classi_Y_location_for_train_input = os.path.join(ingestion_config.get_classi_train_data_dir(), 'train_target.csv')

            classi_X_test_input_location = os.path.join(ingestion_config.get_classi_test_data_dir(),"test_input.csv")
            classi_Y_test_input_location = os.path.join(ingestion_config.get_classi_test_data_dir(), "test_target.csv")


            # using r join since pandas only takes raw string on train test

            classi_X_location_for_train_input = r"\\".join(classi_X_location_for_train_input.split("\\"))   

            classi_Y_location_for_train_input = r"\\".join(classi_Y_location_for_train_input.split("\\"))



            # using r join since pandas only takes raw string on test test

            classi_X_test_input_location = r"\\".join(classi_X_test_input_location.split("\\"))
            classi_Y_test_input_location = r"\\".join(classi_Y_test_input_location.split("\\"))

            classi_X_train.to_csv( classi_X_location_for_train_input)
            classi_y_train.to_csv(classi_Y_location_for_train_input)

            classi_X_test.to_csv( classi_X_test_input_location)
            classi_y_test.to_csv(classi_Y_test_input_location)
            logging.info('Training and testing data seperated successfully for Regression')



            
            data_ingestion_output = data_ingestion_entity(reg_train_input_loc=reg_X_location_for_train_input,
            reg_train_target_loc=reg_Y_location_for_train_input , reg_test_input_loc=reg_X_test_input_location,
            reg_test_target_loc=reg_Y_test_input_location, classi_train_input_loc=classi_X_location_for_train_input,
            classi_train_target_loc=classi_Y_location_for_train_input , classi_test_input_loc=classi_X_test_input_location,
            classi_test_target_loc=classi_Y_test_input_location, complete_csv = self.csv_conversion() )
            return data_ingestion_output


        except Exception as e:
            raise ForestFireException(e, sys) from e


    
    def initiate_data_ingestion(self):
        self.creating_dir()
        self.downloading_files()
        self.csv_conversion()
        #self.splitting_data_saving_to_dir()
        return (self.reg_classi_splitting_data_saving_to_dir())

'''

obj = data_ingestion_component(download_url)
obj.initiate_data_ingestion()

 ''' 

 


       
  
 
    


