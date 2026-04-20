import pandas as pd
import random
import os

def get_random_occupation(education_level, file_path="ISCO-08 -88 EN Index.xlsx"):
    """
    Return result based on education level: 
    Return the level directly if it is a specific in-progress/unfinished education level, 
    otherwise randomly select an occupation from the Excel file.
    
    Parameters:
        education_level: Education level
        file_path: Path to the Excel file
        
    Returns:
        Return the education level if it matches the specific list, 
        otherwise return a randomly selected occupation name
    """
    # Define the list of education levels to return directly
    in_progress_levels = [
        "Primary school (incomplete)",
        "Lower secondary school (attending)",
        "Upper secondary school (attending)",
        "Vocational high school (attending)",
        "Technical school (attending)",
        "Associate degree (in progress)",
        "Bachelor's degree (in progress)",
        "Master's degree (in progress)",
        "Doctoral degree (in progress)"
    ]
    
    # Check if the education level is in the specific list
    if education_level in in_progress_levels:
        return True, education_level
    
    # Otherwise execute the original logic and return a random occupation
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read the Excel file, assuming the first row is the header
        df = pd.read_excel(file_path)
        
        # Check if required columns exist
        required_columns = ['ISCO-08', 'ISCO-88', 'English title']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column in Excel file: {col}")
        
        # Randomly select one row
        random_row = df.sample(n=1).iloc[0]
        
        # Return the occupation name
        return True, random_row['English title']
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return False, None

if __name__ == "__main__":
    # Example education level input, can be modified for testing
    test_education = "Bachelor's degree (in progress)"
    # Excel file path, please modify according to the actual situation
    excel_file = "ISCO-08 -88 EN Index.xlsx"
    
    # Get the result
    result = get_random_occupation(test_education, excel_file)
    
    # Print the result
    if result:
        print(f"Result: {result}")