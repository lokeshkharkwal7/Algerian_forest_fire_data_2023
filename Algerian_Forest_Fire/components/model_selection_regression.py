from Algerian_Forest_Fire.configuration.Algerian_forest_config import model_selection_config
from Algerian_Forest_Fire.exception import ForestFireException
from Algerian_Forest_Fire.constants.constants import *
from Algerian_Forest_Fire.components.data_transformation import data_transformation
from Algerian_Forest_Fire.entity.entity import model_selection_entity
from utility.util import *
from Algerian_Forest_Fire.logger import logging 
import numpy as np
import pandas as pd
import os , pickle , shutil, sys
from sklearn.metrics import mean_squared_error
import pickle
import math
import shutil
import os , sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import importlib
from sklearn.metrics import recall_score

from datetime import datetime

selection_config = model_selection_config()

class model_selection:
    def __init__(self, x_train, y_train, x_test, y_test,x_train_classi,y_train_classi,x_test_classi,
    y_test_classi ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_train_classi = x_train_classi
        self.y_train_classi = y_train_classi
        self.x_test_classi = x_test_classi
        self.y_test_classi = y_test_classi
  
   # Reading all of the datas
   # Reading all the models present in the model yaml 
   # perform grid search cv and using all of the parameters present in config.yaml
   
    def combine_model_selection(self):


        try:

            logging.info('Initiating model Selection')

            # Initiating values/buckets:
            reg_model_performance_test = []
            reg_model_performance_final = []
            reg_fitting_status_= []
            reg_score_training_= []
            reg_respective_parameters_= []
            reg_model_name_= []
            reg_module_address=[]
            reg_pickle_storage = []

            # creating a directory to save all the pric files present in the model

            dir = root_dir

            directory = os.path.join(dir , selection_config.get_model_dir_reg(),current_time)
            os.makedirs(directory, exist_ok = True)

            shutil.rmtree(directory)
            os.makedirs(directory, exist_ok = True)


            # Reading the x_train_transformed, y_train, x_test_transformed, y_test

            x_train_transformed = pd.read_csv(self.x_train)
            print('info about X_train_transformed')
            print(x_train_transformed.info())
            y_train = pd.read_csv(self.y_train)
            y_train = pd.DataFrame(y_train['Temperature'])
            print('info about Y_train_transformed')
            print(y_train.info())

            x_test_transformed= pd.read_csv(self.x_test)
            print("Y test info")
            print(x_test_transformed.info())
            y_test= pd.read_csv(self.y_test)
            y_test = pd.DataFrame(y_test['Temperature'])
            print("Y test info")
            print(y_test.info())

            #  Removing Unnamed: 0 from our transformed data frames
            try:

                for i in [x_train_transformed,x_test_transformed,y_test,y_train]:
                    i.drop(columns = 'Unnamed: 0', inplace = True)
            except Exception as e:
                print(e)
            
            #  reading config files from model.config

            model_data = read_yaml('model_regression.yaml')
            model_data_information = model_data['models']

            logging.info(f'Taking all models and there information from the config file\n {model_data_information}')

            

            # Checking underfitting and overfitting

                # splitting the x_train transformed into split_train_x__transformed and  split_train_y_transformed  

            split_x_train,split_x_test , split_y_train , split_y_test = train_test_split(x_train_transformed,y_train, test_size = 0.33 , random_state = 42)
        
        except Exception as e:
            raise ForestFireException(e, sys) from e 

         # training the model with the splitting datas
         # Testing the datas with the same model
         # print the model
         # calculate the difference


        for i in model_data_information:

            try:
                model_imfo = i
                model_name = model_imfo['name']
                model_parameteres = model_imfo['parameters']

                logging.info(f'Checking Overfitting for {model_name}')

                # dynamically importing the model_name 
                library = importlib.import_module("sklearn.ensemble" if model_name in ["RandomForestRegressor", "GradientBoostingRegressor"] else ("sklearn.linear_model" if model_name in ['Ridge','LinearRegression'] else "sklearn.svm"))
                model_class = getattr(library, model_name)
                final_model = model_class()


                # using grid_search_cv to find out the best parameters
                # Creating an object of gridsearchcv

                grid_search_obj = GridSearchCV(final_model,model_parameteres)
                # Fitting this object to our train and test data 
                grid_search_obj.fit(split_x_train,split_y_train)
                best_parameters = grid_search_obj.best_params_
                y_pred_test = grid_search_obj.predict(split_x_test)
                y_pred_final = grid_search_obj.predict(x_test_transformed)
                score_training = r2_score(split_y_test, y_pred_test,force_finite=False)
                score_testing = r2_score(y_test ,y_pred_final,force_finite=False)
                fitting_differnce = abs(score_testing)-abs(score_training)
                if abs(fitting_differnce)<selection_config.get_fitting_score():
                    fitting_status = 'Proper Fit'
                else:
                    fitting_status = 'Overfitted'
                logging.info(f'Fitting status {fitting_status}')
                logging.info(f'Model_score {score_training}')

                squared_error = mean_squared_error(y_pred_final, y_test)
                squared_error=math.sqrt(squared_error)
                squared_error_z = mean_squared_error(split_y_test, y_pred_test)

                squared_error_z=math.sqrt(squared_error)

                # abs will convert any negative value to positive value
                diff = abs(squared_error_z)-abs(squared_error)
                file_name = f'{model_name}.pkl'  

                with open(file_name,'wb') as f:
                    pickle.dump(grid_search_obj,f)

                # moving the pickel file to model container

                logging.info('Saving file to the Model Container directory')
                shutil.move(file_name , directory) 
                reg_pickle_location = os.path.join(directory, file_name)
                reg_pickle_storage.append(reg_pickle_location)

           # Appending the useful information to the respective list
                
                reg_model_name_.append(model_name)
                reg_respective_parameters_.append(best_parameters)
                reg_score_training_.append(score_training)

                reg_model_performance_test.append(score_testing)

                reg_fitting_status_.append(fitting_status)
                reg_module_address.append(model_class)

            except Exception as e:
                raise ForestFireException(e, sys) from e               
        print(f'Testing Score for the models are {reg_model_performance_test}')

 
    
############################ FOR CLASSIFICATION ####################################

        try:

            logging.info('Initiating model Selection')

            # Initiating values/buckets:
            classi_model_performance_test = []
            classi_model_performance_final = []
            classi_fitting_status_= []
            classi_score_training_= []
            classi_respective_parameters_= []
            classi_model_name_= []
            classi_module_address=[]
            classi_pickle_storage = []

            # creating a directory to save all the pric files present in the model

            dir = root_dir

            directory = os.path.join(dir , selection_config.get_model_dir_classifiction(),current_time)
            os.makedirs(directory, exist_ok = True)


            # Reading the x_train_transformed, y_train, x_test_transformed, y_test

            x_train_transformed = pd.read_csv(self.x_train_classi)
            print('info about X_train_transformed')
            print(x_train_transformed.info())
            y_train = pd.read_csv(self.y_train_classi)
            y_train_transformed = pd.DataFrame(y_train['Classes  '])
            print('info about Y_train_transformed')
            print(y_train.info())

            x_test_transformed= pd.read_csv(self.x_test_classi)
            print("Y test info")
            print(x_test_transformed.info())
            y_test= pd.read_csv(self.y_test_classi)
            y_test_transformed = pd.DataFrame(y_test['Classes  '])
            print("Y test info")
            print(y_test.info())
            print('done asfldkjsfkldjsflk;adsjf; asdjfasklf;')

            #  Removing Unnamed: 0 from our transformed data frames
            try:

                for i in [x_train_transformed,x_test_transformed,y_test,y_train]:
                    i.drop(columns = 'Unnamed: 0', inplace = True)
            except Exception as e:
                print(e)


                
            
            #  reading config files from model.config

            model_data = read_yaml('model_classification.yaml')
            model_data_information = model_data['models']

            logging.info(f'Taking all models and there information from the config file\n {model_data_information}')

            

            # Checking underfitting and overfitting

                # splitting the x_train transformed into split_train_x__transformed and  split_train_y_transformed  

            split_x_train,split_x_test , split_y_train , split_y_test = train_test_split(x_train_transformed,y_train_transformed, test_size = 0.33 , random_state = 42)
        
        except Exception as e:
            raise ForestFireException(e, sys) from e 

         # training the model with the splitting datas
         # Testing the datas with the same model
         # print the model
         # calculate the difference


        for i in model_data_information:

            try:
                print('Please check the value opijdfalsjdf;l')
                print(split_x_train.info())
                model_imfo = i
                model_name = model_imfo['name']
                model_parameteres = model_imfo['parameters']

                logging.info(f'Checking Overfitting for {model_name}')

                # dynamically importing the model_name 
                library = importlib.import_module("sklearn.ensemble" if model_name in ["RandomForestClassifier", "GradientBoostingClassifier"] else ("sklearn.linear_model" if model_name in ['Ridge','LogisticRegression'] else "sklearn.svm"))
                model_class = getattr(library, model_name)
                final_model = model_class()


                # using grid_search_cv to find out the best parameters
                # Creating an object of gridsearchcv

                grid_search_obj = GridSearchCV(final_model,model_parameteres)
                # Fitting this object to our train and test data 
                grid_search_obj.fit(split_x_train,split_y_train)
                best_parameters = grid_search_obj.best_params_
                y_pred_test = grid_search_obj.predict(split_x_test)
                y_pred_final = grid_search_obj.predict(x_test_transformed)

                # testing ( peformance matrix : recall )
                score_training = recall_score(split_y_test,y_pred_test)   
                score_testing = recall_score(y_test_transformed ,y_pred_final)
                fitting_differnce = abs(score_testing)-abs(score_training)
                if abs(fitting_differnce)<selection_config.get_fitting_score():
                    fitting_status = 'Proper Fit'
                else:
                    fitting_status = 'Overfitted'
                logging.info(f'Fitting status {fitting_status}')
                logging.info(f'Model_score {score_training}')
 
 
                file_name = f'{model_name}.pkl'  

                with open(file_name,'wb') as f:
                    pickle.dump(grid_search_obj,f)

                # moving the pickel file to model container

                logging.info('Saving file to the Model Container directory')
                shutil.move(file_name , directory) 
                pickle_location = os.path.join(directory, file_name)
                classi_pickle_storage.append(pickle_location)

           # Appending the useful information to the respective list
                
                classi_model_name_.append(model_name)
                classi_respective_parameters_.append(best_parameters)
                classi_score_training_.append(score_training)

                classi_model_performance_test.append(score_testing)

                classi_fitting_status_.append(fitting_status)
                classi_module_address.append(model_class)

            except Exception as e:
                raise ForestFireException(e, sys) from e               
        print(f'Testing Score for the models are {classi_model_performance_test}')

        model_selection_output = model_selection_entity(reg_model_name=reg_model_name_ , reg_respective_parameters=reg_respective_parameters_,
        reg_score_training=reg_score_training_,reg_fitting_status=reg_fitting_status_,reg_module_address=reg_module_address,reg_pickle_storage_loc = reg_pickle_storage,
        classi_model_name=classi_model_name_ , classi_respective_parameters=classi_respective_parameters_,
        classi_score_training=classi_score_training_,classi_fitting_status=classi_fitting_status_,classi_module_address=classi_module_address,classi_pickle_storage_loc = classi_pickle_storage)

        logging.info(f'Model Selection Completed \n Information: {model_selection_output}')
        return model_selection_output

'''
X_reg_train_trans='C:\\Users\\Lokesh\\Desktop\\Ineurone\\Project\\Algerian Forest Fire Full\\artifact\\2023-17-09_02-18-28\\Reg_Transformed\\X_train_transformed.csv'
y_reg_train_trans='C:\\Users\\Lokesh\\Desktop\\Ineurone\\Project\\Algerian Forest Fire Full\\artifact\\2023-17-09_02-18-28\\Reg_Transformed\\y_train_transformed.csv'
X_reg_test_trans='C:\\Users\\Lokesh\\Desktop\\Ineurone\\Project\\Algerian Forest Fire Full\\artifact\\2023-17-09_02-18-28\\Reg_Transformed\\X_test_transformed.csv'
y_reg_test_trans='C:\\Users\\Lokesh\\Desktop\\Ineurone\\Project\\Algerian Forest Fire Full\\artifact\\2023-17-09_02-18-28\\Reg_Transformed\\y_test_transformed.csv'
z_score = 'C:\\\\Users\\\\Lokesh\\\\Desktop\\\\Ineurone\\\\Project\\\\Algerian Forest Fire Full\\\\Z_scaled_obj\\\\Z_regression.pkl'
pca = 'C:\\\\Users\\\\Lokesh\\\\Desktop\\\\Ineurone\\\\Project\\\\Algerian Forest Fire Full\\\\Transformed_obj\\\\regression.pkl'
X_classi_train_trans='C:\\Users\\Lokesh\\Desktop\\Algerian_forest_fire_data_2023\\artifact\\2023-18-09_18-37-31\\Classi_Transformed\\X_train_transformed.csv'
y_classi_train_trans='C:\\Users\\Lokesh\\Desktop\\Algerian_forest_fire_data_2023\\artifact\\2023-18-09_18-37-31\\Classi_Transformed\\y_train_transformed.csv'
X_classi_test_trans='C:\\Users\\Lokesh\\Desktop\\Algerian_forest_fire_data_2023\\artifact\\2023-18-09_18-37-31\\Classi_Transformed\\X_test_transformed.csv'
y_classi_test_trans='C:\\Users\\Lokesh\\Desktop\\Algerian_forest_fire_data_2023\\artifact\\2023-18-09_18-37-31\\Classi_Transformed\\y_test_transformed.csv'
        

 
obj = model_selection(X_reg_train_trans,y_reg_train_trans,X_reg_test_trans,y_reg_test_trans, 
X_classi_train_trans,y_classi_train_trans,X_classi_test_trans,y_classi_test_trans,'z_score' , 'pca')
print(obj.for_regression())

'''