import sys  # to have information about the exception raised
import logging
from logger import *

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] with error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))
    
    return error_message

class CustomException(Exception):
    
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        logging.error(self.error_message)
        
    def __str__(self) -> str:
        return self.error_message
    
    def __repr__(self) -> str:
        return self.error_message
    
    def __unicode__(self) -> str:
        return self.error_message
    
    # Many will be added based on our code needs.
    
 
 
# to test if we are correctly caring about the exception raised and if they appear in the log file.   
# if __name__ == "__main__":
    
#     try:
#         a = "amine" + 12
#     except Exception as e:
#         logging.info("Can't add string and integer.")
#         raise CustomException(e, sys)