#how data inegstion will happen????????????
#may be get from database or some local file 
#then split the data into train, test
#return the data path- so it can be use by other package or module

#return the data
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os
import sys
import logging
import pandas as pd
from exception import CustomException

@dataclass
class data_ingestion_config:
    train_data_path:str = os.path.join('artifect','train.csv')
    test_data_path:str = os.path.join('artifect', 'test.csv')
    raw_data_path:str = os.path.join('artifect','raw.csv'),
    
class data_ingestion:
    def __int__(self):
        self.data_ingestion_config = data_ingestion_config()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            #get data from database
            ingest_data = pd.read_csv('database/Reddit_Data.csv')
            logging.info('Read the dataset as dataframe')
            
            #ensures directory exists            
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)

            ingest_data.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)
            
            #train_test_split
            train_set, test_set = train_test_split(ingest_data, test_size=0.2,random_state=42) 
            
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False, header=True)
            
            return( self.data_ingestion_config.train_data_path , self.data_ingestion_config.test_data_path)
            
        except CustomException as e:
            raise CustomException(e, sys)
    
if __name__ == "__main__":
    obj = data_ingestion()
    train_set,test_set = obj.initiate_data_ingestion()