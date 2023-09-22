# Importing the information from the previous component
# finding out the indexes of status of underfit and remove all the contents
# find out the index with the best score 
# Get the best names and its respective parameters
# import the file and create a pickle out of it

from Algerian_Forest_Fire.constants.constants import *
from Algerian_Forest_Fire.configuration.Algerian_forest_config import *
from Algerian_Forest_Fire.entity.entity import *
from Algerian_Forest_Fire.exception import ForestFireException
from Algerian_Forest_Fire.logger import logging
import pickle
import numpy as np
from datetime import datetime
import shutil
import os,sys
from utility.util import *

from sklearn.pipeline import Pipeline

pusher_config = model_pusher_config()
  
path = os.path.join(root_dir, 'Model_container\08_09_23_33_22_\RandomForestRegressor.pkl')

data = model_selection_entity(reg_model_name=['RandomForestRegressor', 'GradientBoostingRegressor', 'SVR', 'Ridge'], reg_respective_parameters=[{'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 100}, {'learning_rate': 0.3, 'n_estimators': 200}, {'kernel': 'linear'}, {'alpha': 0.1}], 
reg_score_training=[0.6169133740659748, 0.5908773678977837, 0.5806408232680369, 0.5732565672528152], reg_fitting_status=['Proper Fit', 'Overfitted', 'Proper Fit', 'Overfitted'], reg_module_address=[1,2,3,4], reg_pickle_storage_loc=['C:\\Users\\Lokesh\\Desktop\\Algerian_forest_fire_data_2023\\MODEL_CONTAINER_REGRESSION\\2023-19-09_18-33-29\\RandomForestRegressor.pkl', 'C:\\Users\\Lokesh\\Desktop\\Algerian_forest_fire_data_2023\\MODEL_CONTAINER_REGRESSION\\2023-19-09_18-33-29\\GradientBoostingRegressor.pkl', 'C:\\Users\\Lokesh\\Desktop\\Algerian_forest_fire_data_2023\\MODEL_CONTAINER_REGRESSION\\2023-19-09_18-33-29\\SVR.pkl', 'C:\\Users\\Lokesh\\Desktop\\Algerian_forest_fire_data_2023\\MODEL_CONTAINER_REGRESSION\\2023-19-09_18-33-29\\Ridge.pkl'], 
classi_model_name=['RandomForestClassifier', 'LogisticRegression', 'GradientBoostingClassifier'], classi_respective_parameters=[{'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 3}, {'C': 1.0, 'solver': 'lbfgs'}, {'learning_rate': 0.1, 'max_depth': 3}], classi_score_training=[0.9743589743589743, 1.0, 0.9743589743589743], classi_fitting_status=['Overfitted', 'Overfitted', 'Overfitted'], classi_module_address=['1','2','3','4'], 
classi_pickle_storage_loc=['C:\\Users\\Lokesh\\Desktop\\Algerian_forest_fire_data_2023\\MODEL_CONTAINER_CLASSIFICATION\\2023-19-09_18-33-29\\RandomForestClassifier.pkl', 'C:\\Users\\Lokesh\\Desktop\\Algerian_forest_fire_data_2023\\MODEL_CONTAINER_CLASSIFICATION\\2023-19-09_18-33-29\\LogisticRegression.pkl', 'C:\\Users\\Lokesh\\Desktop\\Algerian_forest_fire_data_2023\\MODEL_CONTAINER_CLASSIFICATION\\2023-19-09_18-33-29\\GradientBoostingClassifier.pkl'])


class model_pusher:
    def __init__(self, reg_output_list):
        self.REG_output_list = reg_output_list
 
    def get_best_model(self) :

        try:

            logging.info('Selecting Best Model for Regression')


            # getting the score training from the output list
            score_training = self.REG_output_list[2]
            best_model_loc = (score_training.index(max (score_training)))
            best_pickle_file_loc = self.REG_output_list[5][best_model_loc]
            best_model_score = self.REG_output_list[2][best_model_loc]
            best_model_name = self.REG_output_list[0][best_model_loc]
            best_model_param= self.REG_output_list[1][best_model_loc]

            # saving all of the data to a dictionary

            model_info_to_yaml = { f'Model{datetime.now().strftime("%d_%m_%H_%M_%S_")}':
            { 'Model_name':best_model_name,
                'Model_score':float(best_model_score),
                'Pickle_Location':best_pickle_file_loc,
                'best_model_parameters':best_model_param}
            }

            logging.info(f'Best Model found\n {model_info_to_yaml}')

            

            print(model_info_to_yaml)

            # Sending the above dictionary to the yaml file

            model_entry_to_yaml(model_info_to_yaml,'Reg_model_status.yaml')

            # selecting the best model_from model_staus.yaml

            best_model_info = get_best_model_from_yaml('Reg_model_status.yaml')

            # getting the pickle file from the best model:

            best_pickle_file_loc = best_model_info['Pickle_Location']

            


            print(best_pickle_file_loc)

            # saving the file to the best model_location
            with open(best_pickle_file_loc , 'rb') as f:
                model = pickle.load(f)

            # Deleting all the models which are present in the best_model_folder



            # transfering this final pickle file to the best_model_directory


 
            directory = os.path.join(root_dir , pusher_config.get_reg_best_model_dir(),current_time)
            os.makedirs(directory, exist_ok = True)
            
            # moving this file to the above directory

            shutil.copy(best_pickle_file_loc,directory)
            print(f'Directory for best pickle_file {directory}')

           # Opening the z_scaled object 
        
            scaled_obj_file_path = os.path.join(root_dir, 'Z_scaled_obj','Z_regression_.pkl')
            with open(scaled_obj_file_path, 'rb') as file:
                scaled_object = pickle.load(file)

            
            # Now  open the PCA transformed object

            reg_transformed_obj_filepath = os.path.join(root_dir,'Transformed_obj','regression_.pkl')
            with  open(reg_transformed_obj_filepath, 'rb') as f:
                trasnformed_obj = pickle.load(f)

            
            # Combing all three pickle files  into a single pipeline and saving it to a new directory 

            logging.info('Creating a final pipeline and its pickle file which includes steps from Data Transformation and Best Model Selection')

            final_pipeline = Pipeline([
                ('z_score_scaling' , scaled_object),
                ('transformation', trasnformed_obj),
                ('final_model', model)

            ])
            #print(final_pipeline.predict([[1,6,2012,29,57,18,0,65.7,3.4,7.6,1.3,3.4,0.5,0 ]]))   

            # Creating an object of the final pipeline and saving it to the desired location

            final_pipeline_dir_location = os.path.join(root_dir , 'Reg_Final_Pipeline')
            os.makedirs(final_pipeline_dir_location , exist_ok=True)
            final_pipeline_obj_location = os.path.join(final_pipeline_dir_location, 'final_pipeline.pkl')

            # Removing everything that is present in the Final_pipeline_folder

            #shutil.rmtree(final_pipeline_dir_location)

            # Creating pipeline and saving it to the Final_pipeline_folder

            with open(final_pipeline_obj_location, 'wb') as f:
                pickle.dump(final_pipeline, f)

        except Exception as e:
            raise ForestFireException(e, sys) from e

        logging.info(f'Pickle of Final Pipeline created successfully\n loc: {final_pipeline_obj_location}')


        # Loading the pickle file for the output
        try:
            if( final_pipeline_obj_location):
                status = 'Pipeline Completed Successfully'
                logging.info(status)
            
            else:
                status = "Unable to Complete Pipeline"
                logging.info(status)

        except Exception as e:
            raise ForestFireException(e, sys) from e


######################## FOR CLASSIFICATION #########################################################
                    # getting the score training from the output list
        try:
            logging.info('Selecting Best Model for Classification')
            score_training = self.REG_output_list[8] 
            best_model_loc = (score_training.index(max (score_training)))
            best_pickle_file_loc = self.REG_output_list[11][best_model_loc]
            best_model_score = self.REG_output_list[8][best_model_loc]
            best_model_name = self.REG_output_list[6][best_model_loc]
            best_model_param= self.REG_output_list[7][best_model_loc]

            # saving all of the data to a dictionary

            model_info_to_yaml = { f'Model{datetime.now().strftime("%d_%m_%H_%M_%S_")}':
            { 'Model_name':best_model_name,
                'Model_score':float(best_model_score),
                'Pickle_Location':best_pickle_file_loc,
                'best_model_parameters':best_model_param}
            }

            logging.info(f'Best Model found\n {model_info_to_yaml}')

            

            print(model_info_to_yaml)

            # Sending the above dictionary to the yaml file

            model_entry_to_yaml(model_info_to_yaml,'Classi_model_status.yaml')

            # selecting the best model_from model_staus.yaml

            best_model_info = get_best_model_from_yaml('Classi_model_status.yaml')

            # getting the pickle file from the best model:

            best_pickle_file_loc = best_model_info['Pickle_Location']

            


            print(best_pickle_file_loc)

            # saving the file to the best model_location
            with open(best_pickle_file_loc , 'rb') as f:
                model = pickle.load(f)

            # Deleting all the models which are present in the best_model_folder

            # transfering this final pickle file to the best_model_directory

            directory = os.path.join(root_dir , pusher_config.get_classi_best_model_dir(),current_time)
            os.makedirs(directory, exist_ok = True)
            
            # moving this file to the above directory

            shutil.copy(best_pickle_file_loc,directory)
            print(f'Directory for best pickle_file {directory}')

            # importing the scaling object 

            scaling_file_path = os.path.join(root_dir , 'Z_scaled_obj','Z_classification_.pkl')
            with open(scaling_file_path , 'rb') as file:
                scaled_object = pickle.load(file)
            
            # Now  open the transformed object


            transformed_file_path = os.path.join(root_dir , 'Transformed_obj', 'classification_.pkl')
            with open(scaling_file_path , 'rb') as file:
                transformed_object = pickle.load(file)

            
            # Combing all three pickle files  into a single pipeline and saving it to a new directory 

            logging.info('Creating a final pipeline and its pickle file which includes steps from Data Transformation and Best Model Selection')

            final_pipeline = Pipeline([
                ('Z_score scaling', scaled_object),
                ('transformation', trasnformed_obj),
                ('final_model', model)

            ])

            # Creating an object of the final pipeline and saving it to the desired location

            final_pipeline_dir_location = os.path.join(root_dir , 'Classi_Final_Pipeline')
            os.makedirs(final_pipeline_dir_location , exist_ok=True)
            final_pipeline_obj_location = os.path.join(final_pipeline_dir_location, 'final_pipeline.pkl')

            # Removing everything that is present in the Final_pipeline_folder

            #shutil.rmtree(final_pipeline_dir_location)

            # Creating pipeline and saving it to the Final_pipeline_folder

            with open(final_pipeline_obj_location, 'wb') as f:
                pickle.dump(final_pipeline, f)

        except Exception as e:
            raise ForestFireException(e, sys) from e

        logging.info(f'Pickle of Final Pipeline created successfully\n loc: {final_pipeline_obj_location}')


        try:

        # Loading the pickle file for the output
     
            if( final_pipeline_obj_location):
                status = 'Pipeline Completed Successfully'
                logging.info(status)
            
            else:
                status = "Unable to Run Pipeline"
                logging.info(status)

        except Exception as e:
            raise ForestFireException(e, sys) from e 
        logging.info('Pipeling completed Successfully')   
        print('Pipeline completed succesfully')

         

 

'''

obj = model_pusher(data) 
 
output = obj.get_best_model()
 

 
print(f'Output of the data_model is : {output}')
 

 '''
 




 



        
        
        # once getting this in the list format find out the location of the max value

         # once find out the max location location get the pickle file 
        # move it to the location of the best location
         
         
                
                
            

 


 # Print the dynamically allocated variables
 