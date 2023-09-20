import pickle
from flask import Flask, request, render_template
import numpy as np
import os,sys
import pandas as pd
from Algerian_Forest_Fire.exception import ForestFireException
from Algerian_Forest_Fire.pipeline.main_pipeline import main_pipeline
from utility.util import *

# check if the pipeline objects are working correctly or not 
# initiating both model best pipelines on the begning
# adding additional form to the form input ( project input.html)
# updating its output functions of the data  
 
app = Flask(__name__)

# Open our pickle file in which model information is stored.
try:
    model_regression = pickle.load(open('Reg_Final_Pipeline\\final_pipeline.pkl', 'rb'))
    model_classification = pickle.load(open('Classi_Final_Pipeline\\final_pipeline.pkl', 'rb'))
    
except Exception as e:
            raise ForestFireException(e, sys) from e

# This function will execute as our home screen stored in home.html
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

# This function is linked with the html form by the name of predict and
# this will first take up the value from the form and then predict it with
# our model stored in the pickle


@app.route('/run_pipeline')
def run_pipeline():

     try:  

        pipeline = main_pipeline()
        pipeline.initiate_pipeline()

        # getting the info of the latest log file 

        log_path = get_latest_file('logs')

            # Read the content of the text file
   
        with open(log_path, 'r') as file:
            log_content = file.read()

        # Render the content in an HTML template
 
        return render_template('Pipeline_log.html', text_content=log_content)
     except Exception as e:
            raise ForestFireException(e, sys) from e


# showing logs: 
@app.route('/show_logs')
def show_logs():

    log_path = get_latest_file('logs')

            # Read the content of the text file
   
    with open(log_path, 'r') as file:
            log_content = file.read()

        # Render the content in an HTML template
 
    return render_template('Pipeline_log.html', text_content=log_content)
   

         

# for rendering project_input
@app.route('/project_input')
def project_input():

    return render_template('project_input.html')




@app.route('/predict_regressor', methods=['POST'])
def predict_regressor():

    

        data = [float(x) for x in request.form.values()]
        final_feature = np.array(data)

        row_to_predict = [final_feature]
        df=pd.DataFrame(row_to_predict)

        output=model_regression.predict(df)
 
        output =round(output[0],2)

           
        print(output)
         
        prediction_result = f"Predicted Result: {output} Celsius"

        return render_template('result.html', prediction_text=prediction_result)

@app.route('/predict_classification', methods=['POST'])
def predict_classification():

    

        data = [float(x) for x in request.form.values()]
        final_feature = np.array(data)

        row_to_predict = [final_feature]
        df=pd.DataFrame(row_to_predict)

        output=model_classification.predict(df)

        if(output == 1):
            result= "Yes there is a Forest Fire in Algerian Forest"
        elif(output == 0):
            result = 'Relax there is No forest fire in Algerian Forest'
           
        print(output)
         
        prediction_result = f"{result}"

        return render_template('result.html', prediction_text=prediction_result)
 
# Data drift report
@app.route('/data_drift_report')
def data_drift_report():
    return render_template('Data_drift_report/Data_Drift_Report.html')

# for Pandas Profiling before the transformation
@app.route('/data_analysis_report')
def data_analysis_report():
    return render_template('PandasProfiling_before/Pandas_profiling.html')

# for Pandas Profiling after the transformation
@app.route('/data_analysis_transformed_report')
def data_analysis_transformed_report():
    return render_template('Transformed_Data_Analysis/Pandas_Profile.html')

# Creating a file system that contains all the directories containing in the artifact folder
# Define the root directory you want to list
 

# Define the root directory you want to list
root_directory = 'artifact'  # Replace with your directory name

def list_directory_contents(directory):
    # Get a list of all items (files and subdirectories) in the given directory

    try:

        items = os.listdir(directory)

        # Separate files and subdirectories
        files = []
        subdirectories = []

        for item in items:
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                subdirectories.append(item)
            else:
                files.append(item)

        return files, subdirectories

    except Exception as e:
            raise ForestFireException(e, sys) from e

def generate_directory_tree(directory):
    try:
        # Generate a tree structure of the directory and its subfolders
        tree = {}

        for root, dirs, files in os.walk(directory):
            current_node = tree
            for dir_name in os.path.relpath(root, directory).split(os.sep):
                if dir_name not in current_node:
                    current_node[dir_name] = {}
                current_node = current_node[dir_name]

            current_node['FILES'] = files

        return tree
    except Exception as e:
            raise ForestFireException(e, sys) from e

@app.route('/opening_artifact')
def opening_artifact():
    try:

        # Get the requested directory path from the query parameter
        directory = request.args.get('dir', root_directory)

        # Get the absolute path of the requested directory
        current_directory = os.path.dirname(os.path.abspath(__file__))
        directory_absolute_path = os.path.join(current_directory, directory)

        # Ensure the requested directory is within the root directory
        if not directory_absolute_path.startswith(os.path.join(current_directory, root_directory)):
            return "Access denied."

        # List the files and subdirectories in the requested directory
        files, subdirectories = list_directory_contents(directory_absolute_path)

        # Generate a directory tree structure
        directory_tree = generate_directory_tree(directory_absolute_path)

        return render_template('list_directory.html', root_directory=root_directory, directory=directory, subdirectories=subdirectories, directory_tree=directory_tree, files=files)
    except Exception as e:
            raise ForestFireException(e, sys) from e

@app.route('/view_file/<path:file_path>')
def view_file(file_path):

    try:
        # Get the absolute path of the requested file
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_absolute_path = os.path.join(current_directory, file_path)

        # Ensure the requested file is within the root directory
        if not file_absolute_path.startswith(os.path.join(current_directory, root_directory)):
            return "Access denied."

        # Check if the file exists and is a file (not a directory)
        if os.path.exists(file_absolute_path) and os.path.isfile(file_absolute_path):
            return send_file(file_absolute_path)

        # If the file doesn't exist or is not allowed, return a 404 error
        return "File not found or not allowed.", 404

    except Exception as e:
            raise ForestFireException(e, sys) from e



# reading config files for the best model selected:


@app.route('/display_text_file')
def display_text_file():
    try:
        # Read the content of the text file
        with open('Reg_model_status.yaml', 'r') as file:
            text_content = file.read()

        # Render the content in an HTML template
        return render_template('model_status.html', text_content=text_content)
    except Exception as e:
            raise ForestFireException(e, sys) from e



# reading config files for models and its parameters:


@app.route('/models_used')
def models_used():
    # Read the content of the text file
    try:
        with open('model_regression.yaml', 'r') as file:
            text_content = file.read()

        # Render the content in an HTML template
        return render_template('all_models.html', text_content=text_content)
    except Exception as e:
            raise ForestFireException(e, sys) from e


@app.route('/configuration')
def configuration():
    try:
        # Read the content of the text file
        with open('config.yaml', 'r') as file:
            text_content = file.read()

        # Render the content in an HTML template
        return render_template('configuration.html', text_content=text_content)
    except Exception as e:
            raise ForestFireException(e, sys) from e



@app.route('/get_schema')
def get_schema():

    # Read the content of the text file
    try:
        with open('schema.yaml', 'r') as file:
            text_content = file.read()

        # Render the content in an HTML template
        return render_template('schema.html', text_content=text_content)
    except Exception as e:
            raise ForestFireException(e, sys) from e

 

if __name__ == '__main__':
    app.run(debug=True)
 

 
  
