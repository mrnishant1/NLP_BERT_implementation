from dataclasses import dataclass
import os

@dataclass 
class data_preprocessor_config:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

# class DataTransformation: