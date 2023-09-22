from Algerian_Forest_Fire.exception import ForestFireException
from Algerian_Forest_Fire.logger import logging
from Algerian_Forest_Fire.components.data_ingestion import data_ingestion_component
from Algerian_Forest_Fire.components.data_transformation import *
from Algerian_Forest_Fire.components.data_validation import *
from Algerian_Forest_Fire.components.model_pusher_regression import *
from Algerian_Forest_Fire.components.model_selection_regression import *
from Algerian_Forest_Fire.entity.entity import *

class main_pipeline:
    def __init__(self):
        pass
    def initiate_pipeline(self):
     try:

        # Initate Data Ingestion

        data_ingestion = data_ingestion_component()
        ingestion = data_ingestion.initiate_data_ingestion()
        print(ingestion[8])


        # Initiate Data Validation
        validation = data_validation(ingestion[0],ingestion[2],ingestion[8])
        print(validation.initiate_data_validation())

 
        # Initiate Data Transformation

        data_transformation_ = data_transformation(ingestion[0], ingestion[1], ingestion[2],ingestion[3], ingestion[4], ingestion[5], ingestion[6], ingestion[7])
        transformation = data_transformation_.initiate_transformation()
         # Initiate Model Selection 
        selection = model_selection(transformation[4],transformation[5],transformation[6],transformation[7],transformation[8],transformation[9],transformation[10],transformation[11])
        selection_output = selection.combine_model_selection()

        # Initiate Model Pusher

        pusher = model_pusher(selection_output)
        output_pusher = pusher.get_best_model()
        print(output_pusher)
     except Exception as e:
        raise ForestFireException(e,sys) from e 

 
'''
        # Initiate Model Pusher
object = main_pipeline()
print(object.initiate_pipeline())
 '''