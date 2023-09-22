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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import shutil
from pandas_profiling import ProfileReport
from utility.util import *
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from pandas_profiling import ProfileReport

import pickle
import matplotlib.pyplot as plt
# removing unwanted elements from the data frame

transformation_config = data_transformation_config()

class data_transformation:
    def __init__(self,reg_x_train,reg_y_train, reg_x_test,reg_y_test,classi_x_train,classi_y_train,classi_x_test,classi_y_test):
        # Reg
        self.reg_x_train = reg_x_train
        self.reg_x_test = reg_x_test
        self.reg_y_train = reg_y_train
        self.reg_y_test = reg_y_test

        # Classi
        self.classi_x_train = classi_x_train
        self.classi_x_test = classi_x_test
        self.classi_y_train = classi_y_train
        self.classi_y_test = classi_y_test

    # creating dataframes for all from the above location to a dictionary

    def get_dataframes(self):

     try:
        df_loc = { 
            'X_reg_train':self.reg_x_train, 
            'X_reg_test':self.reg_x_test,
            'X_classi_train':self.classi_x_train,
            'X_classi_test':self.classi_x_test,
            'y_reg_train':self.reg_y_train,
            'y_reg_test':self.reg_y_test,
            'y_classi_train':self.classi_y_train,
            'y_classi_test':self.classi_y_test
        }

        dataframes = {}
        for key, items in df_loc.items():
            df = pd.read_csv(items)
            df.drop(columns = ['Unnamed: 0'], inplace = True)
            dataframes[key]=df
        return dataframes
     except Exception as e:
        raise ForestFireException(e,sys) from e


 


    def initiate_transformation(self):

     
        pca_obj=[]
        z_scale_obj = []

        def get_pca_graph(data_frame):
            try:

                scaler = StandardScaler()
                z_scored_data = scaler.fit_transform(data_frame)
                pca = PCA()
                pca.fit_transform(data_frame)
                plt.figure()
                plt.plot( np.cumsum(pca.explained_variance_ratio_))
                plt.xlabel('No of columns')
                plt.ylabel('Explained varience ratio')
                saving_pca_graph = os.path.join(root_dir, 'PCA')
                os.makedirs(saving_pca_graph , exist_ok=True)
                save_path = os.path.join(saving_pca_graph, 'PCA.png')
                plt.savefig(save_path)
                plt.show()
            except Exception as e:
                raise ForestFireException(e,sys) from e

        def pca_transform(dataframe , nd):
         try:
            n_comp = transformation_config.get_pca_estimator()
            pca = PCA(n_components= n_comp)
            transformed_data = pca.fit_transform(dataframe)
            loc = f'{nd}.pkl'
            
            with open(loc, 'wb') as file:
                  pickle.dump(pca, file)
            pca_obj.append(loc)        
            return transformed_data
         except Exception as e:
            raise ForestFireException(e,sys) from e

        def Z_scaling(dataframe, nd):
         try:
            scaler = StandardScaler()
            dataframe_scaled = scaler.fit_transform(dataframe)
            path = f'{nd}.pkl'
            with open (path , 'wb') as file:
                pickle.dump(scaler, file)
            z_scale_obj.append(path)

            return dataframe_scaled
         except Exception as e:
             raise ForestFireException(e,sys) from e

        try:
        


            dataframes = self.get_dataframes()
            logging.info('Initiating Data Transformation')
            logging.info('Standardizing the data into z score using Standard Scaler')       
            logging.info('Transorming other features of the data frame')
            

            # Z_score Scaling of the data 
            X_regression_test = Z_scaling(dataframes['X_reg_test'], 'Z_regression')

            X_regression_train = Z_scaling(dataframes['X_reg_train'], 'Z_regression_')

            X_classification_test= Z_scaling(dataframes['X_classi_test'], 'Z_classification')
            X_classification_train= Z_scaling(dataframes['X_classi_train'], 'Z_classification_')

            get_pca_graph(X_classification_train)     

            logging.info(f'Initiating Dimensity reduction using PCA with N_components set to: {transformation_config.get_pca_estimator()}')
            # Transforming regression 

            transformed_test_regression_df = pd.DataFrame(pca_transform(X_regression_test,'regression'))
            transformed_train_regression_df = pd.DataFrame(pca_transform(X_regression_train,'regression_'))

            # Transformed classification
            transformed_test_classification_df = pd.DataFrame(pca_transform(X_classification_test, 'classification')) 
            transformed_train_classification_df = pd.DataFrame(pca_transform(X_classification_train, 'classification_')) 
            
            # Moving the data to the respective artifact directories 

            logging.info('Data Transformed successfuly moving the transformed data into respective directories')

            reg_transformed_dir = os.path.join(time_stamp_dir_loc, 'Reg_Transformed')
            os.makedirs(reg_transformed_dir, exist_ok=True)
            classi_transformed_dir = os.path.join(time_stamp_dir_loc, 'Classi_Transformed')
            os.makedirs(classi_transformed_dir, exist_ok=True)
            # creating filepath for all the transformed dataframes that were stored in the dataframes dictionary
            #For y
        
            y_regression_train_loc = os.path.join(reg_transformed_dir, 'y_train_transformed.csv')
            y_regression_test_loc = os.path.join(reg_transformed_dir, 'y_test_transformed.csv')
            
            y_classification_train_loc=os.path.join(classi_transformed_dir, 'y_train_transformed.csv')
            y_classification_test_loc=os.path.join(classi_transformed_dir, 'y_test_transformed.csv')

            # For X
            X_regression_train_loc = os.path.join(reg_transformed_dir, 'X_train_transformed.csv')
            X_regression_test_loc = os.path.join(reg_transformed_dir, 'X_test_transformed.csv')
            X_classification_train_loc = os.path.join(classi_transformed_dir, 'X_train_transformed.csv')
            X_classification_test_loc = os.path.join(classi_transformed_dir, 'X_test_transformed.csv')
            # transfering CSV files to the above mentioned location
            # For X
            transformed_train_classification_df.to_csv(X_classification_train_loc)
            transformed_test_classification_df.to_csv(X_classification_test_loc)
            transformed_train_regression_df.to_csv(X_regression_train_loc)
            transformed_test_regression_df.to_csv(X_regression_test_loc)

            #For Y ( Transformation is not required)

            dataframes['y_reg_train'].to_csv(y_regression_train_loc)
            dataframes['y_reg_test'].to_csv(y_regression_test_loc)

            dataframes['y_classi_train'].to_csv(y_classification_train_loc)
            dataframes['y_classi_test'].to_csv(y_classification_test_loc)

        

            # Moving the transformed obj store in the  pca_obj list
            trans_obj_dir = os.path.join(root_dir , 'Transformed_obj')
            shutil.rmtree(trans_obj_dir)
            os.makedirs(trans_obj_dir, exist_ok=True) 

            shutil.move(pca_obj[1],trans_obj_dir)
            shutil.move(pca_obj[3], trans_obj_dir)
            pca_obj_reg_loc = os.path.join(trans_obj_dir,pca_obj[0])
            pca_obj_classi_loc = os.path.join(trans_obj_dir,pca_obj[1])

            # Moving the Z_scaled obj store in z_scale_obj


            z_trans_dir = os.path.join(root_dir , 'Z_scaled_obj')
            shutil.rmtree(z_trans_dir)
            os.makedirs(z_trans_dir, exist_ok=True)
            shutil.move(z_scale_obj[1],z_trans_dir)
            shutil.move(z_scale_obj[3], z_trans_dir)
            z_obj_reg_loc = os.path.join(z_trans_dir,z_scale_obj[0])
            z_obj_classi_loc = os.path.join(z_trans_dir,z_scale_obj[1])

            # For Front End

            # Using pandas profiling to save the transformed data report 
            report_obj = ProfileReport(transformed_train_regression_df)
            html_path = os.path.join(root_dir,'Templates','Transformed_Data_Analysis')
            shutil.rmtree(html_path)
            os.makedirs(html_path, exist_ok=True)
            file_loc = os.path.join(html_path,'Pandas_Profile.html')
            report_obj.to_file(file_loc)
            artifact_trans_file_path = os.path.join(time_stamp_dir_loc,'Transformed_Data_Analysis')
            os.makedirs(artifact_trans_file_path,exist_ok=True)
            shutil.copy(file_loc , artifact_trans_file_path)
            logging.info('Data Transformation Completed')
    
            



            transformation_output = data_transformation_entity(reg_trans_obj = pca_obj_reg_loc, 
            classi_trans_obj =pca_obj_classi_loc  , 
            z_reg_obj=z_obj_reg_loc ,
            z_classi_obj = z_obj_classi_loc,
            X_reg_train_trans =  X_regression_train_loc,
            y_reg_train_trans = y_regression_train_loc,
            X_reg_test_trans = X_regression_test_loc,
            y_reg_test_trans = y_regression_test_loc,
            
            X_classi_train_trans = X_classification_train_loc,
            y_classi_train_trans =  y_classification_train_loc,
            X_classi_test_trans = X_classification_test_loc,
            y_classi_test_trans = y_classification_test_loc)
            return transformation_output
            
        except Exception as e:
           raise ForestFireException(e,sys) from e
        
 
'''
reg_train_input_loc='C:\\\\Users\\\\Lokesh\\\\Desktop\\\\Ineurone\\\\Project\\\\Algerian Forest Fire Full\\\\artifact\\\\2023-16-09_22-37-21\\\\reg_train_test_data\\\\reg_train_data\\\\train_input.csv'

reg_train_target_loc='C:\\\\Users\\\\Lokesh\\\\Desktop\\\\Ineurone\\\\Project\\\\Algerian Forest Fire Full\\\\artifact\\\\2023-16-09_22-37-21\\\\reg_train_test_data\\\\reg_train_data\\\\train_target.csv'

reg_test_input_loc='C:\\\\Users\\\\Lokesh\\\\Desktop\\\\Ineurone\\\\Project\\\\Algerian Forest Fire Full\\\\artifact\\\\2023-16-09_22-37-21\\\\reg_train_test_data\\\\reg_test_data\\\\test_input.csv' 

reg_test_target_loc='C:\\\\Users\\\\Lokesh\\\\Desktop\\\\Ineurone\\\\Project\\\\Algerian Forest Fire Full\\\\artifact\\\\2023-16-09_22-37-21\\\\reg_train_test_data\\\\reg_test_data\\\\test_target.csv'

classi_train_input_loc='C:\\\\Users\\\\Lokesh\\\\Desktop\\\\Ineurone\\\\Project\\\\Algerian Forest Fire Full\\\\artifact\\\\2023-16-09_22-37-21\\\\classi_train_test_data\\\\classi_train_data\\\\train_input.csv'

classi_train_target_loc='C:\\\\Users\\\\Lokesh\\\\Desktop\\\\Ineurone\\\\Project\\\\Algerian Forest Fire Full\\\\artifact\\\\2023-16-09_22-37-21\\\\classi_train_test_data\\\\classi_train_data\\\\train_target.csv'

classi_test_input_loc='C:\\\\Users\\\\Lokesh\\\\Desktop\\\\Ineurone\\\\Project\\\\Algerian Forest Fire Full\\\\artifact\\\\2023-16-09_22-37-21\\\\classi_train_test_data\\\\classi_test_data\\\\test_input.csv'

classi_test_target_loc='C:\\\\Users\\\\Lokesh\\\\Desktop\\\\Ineurone\\\\Project\\\\Algerian Forest Fire Full\\\\artifact\\\\2023-16-09_22-37-21\\\\classi_train_test_data\\\\classi_test_data\\\\test_target.csv'


obj = data_transformation(reg_train_input_loc,  reg_train_target_loc, reg_test_input_loc, reg_test_target_loc,classi_train_input_loc, classi_train_target_loc, classi_test_input_loc,  classi_test_target_loc  )


print(obj.initiate_transformation() )

'''
 
 

 
  
                 







    # getting single training data 


    # filling null values
    # removing outliers 
    # converting the data into classification and regression
    # creating a function that will create a pca transformation of data and will return the entire dataframe
    # creating a location and saving the transformed data seprately 


 