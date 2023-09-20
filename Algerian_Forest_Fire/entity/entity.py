from collections import namedtuple



# entity for data ingestion 


data_ingestion_entity = namedtuple('data_ingestion_entity' ,['reg_train_input_loc' ,
            'reg_train_target_loc' , 'reg_test_input_loc',
            'reg_test_target_loc' , 'classi_train_input_loc',
            'classi_train_target_loc' , 'classi_test_input_loc',
            'classi_test_target_loc','complete_csv'] )
# entity for data validation 

data_validation_entity = namedtuple('data_validation_entity', ['Validation_status', 'Profile_report_location'])

# entity for data transformation 

data_transformation_entity = namedtuple('data_transformation_entity', [ 'reg_trans_obj' , 'classi_trans_obj' , 'z_reg_obj','z_classi_obj',
'X_reg_train_trans','y_reg_train_trans','X_reg_test_trans','y_reg_test_trans','X_classi_train_trans','y_classi_train_trans'
,'X_classi_test_trans','y_classi_test_trans'])
 
 # Model Selection entity

model_selection_entity = namedtuple("model_selection_entity", ['reg_model_name','reg_respective_parameters',
 'reg_score_training','reg_fitting_status','reg_module_address','reg_pickle_storage_loc','classi_model_name','classi_respective_parameters',
 'classi_score_training','classi_fitting_status','classi_module_address','classi_pickle_storage_loc'])