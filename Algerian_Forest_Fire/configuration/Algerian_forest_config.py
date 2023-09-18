from Algerian_Forest_Fire.constants.constants import *
from datetime import datetime
from utility.util import *




# data Ingestion config 

class data_ingestion_config:

    # for Regression
    def __init__(self):
        pass
    
    def get_url(self):
        return URL

    def get_raw_data_dir_loc(self):

        return os.path.join(time_stamp_dir_loc, 'raw_data')
        

    def get_tgz_file_loc(self):

        raw_data_dir_loc = os.path.join(time_stamp_dir_loc, 'raw_data')
        download_file_location = os.path.join(raw_data_dir_loc, 'Algerian_Forest_Fire.tgz')
        return download_file_location

    def get_csv_file_location(self):
        
        csv_file_location = os.path.join(time_stamp_dir_loc , 'csv_file')
        return csv_file_location

    def get_final_csv_location(self):
        location = self.get_csv_file_location()
        csv_loc = os.path.join(location,'Algerian_forest_fires_dataset_UPDATE.csv')
        return csv_loc





    def get_file_location_reg(self):
        csv_file_location = os.path.join(time_stamp_dir_loc , 'csv_file')
        reg_csv_path = os.path.join(csv_file_location , 'Regression')
        return reg_csv_path
    
    def get_reg_csv_path(self):     
        csv_file_location = os.path.join(time_stamp_dir_loc , 'csv_file')
        file_location_reg = os.path.join(csv_file_location, "Algerian_forest_fires_dataset_UPDATE.csv")
        return file_location_reg

    def get_reg_train_test_data_dir(self):
        reg_train_test_location = os.path.join(time_stamp_dir_loc , 'reg_train_test_data')
        return reg_train_test_location

    def get_reg_train_data_dir(self):
        reg_train_test_location = os.path.join(time_stamp_dir_loc , 'reg_train_test_data')
        reg_train_data_location = os.path.join(reg_train_test_location, "reg_train_data")
        return reg_train_data_location

    def get_reg_test_data_dir(self):
        reg_train_test_location = os.path.join(time_stamp_dir_loc , 'reg_train_test_data')
        reg_test_data_location = os.path.join(reg_train_test_location, "reg_test_data")
        return reg_test_data_location


                # for classification 

    def get_classi_train_test_data_dir(self):
        reg_train_test_location = os.path.join(time_stamp_dir_loc , 'classi_train_test_data')
        return reg_train_test_location

    def get_classi_train_data_dir(self):
        reg_train_test_location = os.path.join(time_stamp_dir_loc , 'classi_train_test_data')
        reg_train_data_location = os.path.join(reg_train_test_location, "classi_train_data")
        return reg_train_data_location

    
    def get_classi_test_data_dir(self):
        reg_train_test_location = os.path.join(time_stamp_dir_loc , 'classi_train_test_data')
        reg_test_data_location = os.path.join(reg_train_test_location, "classi_test_data")
        return reg_test_data_location 
        
    def get_z_threshold(self):
        return Z_THRESHOLD



# data Ingestion validation  


class data_validation_config:
    def __init__(self):
        pass

    def get_columns_name(self):

        trans_data = read_yaml('schema.yaml')
        columns_data = trans_data['column names']
        return columns_data['names']

    def get_float_data_types_columns(self):

        trans_data = read_yaml('schema.yaml')
        columns_data = trans_data['column float dtypes']
        return columns_data['names']

    def get_object_data_types_columns(self):
        
        trans_data = read_yaml('schema.yaml')
        columns_data = trans_data['column object dtypes']
        return columns_data['names']

 # Configuration for data validation        
 
class data_transformation_config:
    def __init__(self):
        pass


    def get_pca_estimator(self):
        return PCA_EST


# configuration for model Selection 

class model_selection_config:
    def __init__(self):
        pass
    def get_model_dir_reg(self):
        return MODEL_CONTAINER_DIR_REG
    def get_model_dir_classifiction(self):
        return MODEL_CONTAINER_DIR_CLASSI
    def get_fitting_score(self):
        return FITTING_SCORE






       







    




           






 
    


