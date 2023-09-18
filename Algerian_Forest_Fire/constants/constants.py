# For Data Ingestion 
from utility.util import *
from datetime import datetime
import os 
 
root_dir = os.getcwd()

artifact_dir  = os.path.join(root_dir, 'artifact')

current_time = datetime.now().strftime('20%y-%d-%m_%H-%M-%S')

time_stamp_dir_loc = os.path.join(artifact_dir , current_time)


config_data = read_yaml('config.yaml')

URL = config_data['data_ingestion']['url']

Z_THRESHOLD = config_data['data_transformation']['z_threshold']
PCA_EST = config_data['data_transformation']['pca_estimator']
 
MODEL_CONTAINER_DIR_REG = config_data['model_selection']['model_container_dir_reg']
FITTING_SCORE = config_data['model_selection']['fitting_score']
MODEL_CONTAINER_DIR_CLASSI = config_data['model_selection']['model_container_dir_classi']

print(MODEL_CONTAINER_DIR_CLASSI)
print(MODEL_CONTAINER_DIR_REG)




 
 



 