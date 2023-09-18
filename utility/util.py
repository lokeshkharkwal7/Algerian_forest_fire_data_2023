# reading yaml file 
import yaml


def read_yaml(file_path) -> dict:
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        return data

# writing yaml file

def write_yaml(data , file_path):
    with open(file_path , 'w') as file:
        yaml.dump(data , file , default_flow_style=False)



