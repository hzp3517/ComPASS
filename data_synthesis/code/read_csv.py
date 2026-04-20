import csv
import json

def csv_to_utterance_dict(csv_file_path):
    """
    Converts a CSV file containing JSON strings into a Python dictionary.
    Uses 'sample_id' as the key and the parsed JSON 'generation' data as the value.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        
    Returns:
        dict: Dictionary with integer sample IDs as keys and parsed JSON data as values
    """
    result_dict = {}
    
    # Open CSV file with utf-8 encoding to handle special characters
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        # Read CSV using headers as dictionary keys
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            # Extract sample_id and convert to integer for better usability
            sample_id = int(row['sample_id'])
            try:
                # Parse the JSON string in the 'generation' column
                generation_dict = json.loads(row['generation'])
            except json.JSONDecodeError as e:
                # Handle JSON parsing errors
                print(f"Warning: (sample_id: {sample_id}) Failed to parse JSON: {e}")
                print(f"Raw data snippet: {row['generation'][:100]}...")  # Print first 100 chars
                generation_dict = None  # Can also use empty dict {}

            # Store parsed data in the result dictionary
            result_dict[sample_id] = generation_dict
    
    return result_dict

# ------------------- Example Usage -------------------
if __name__ == "__main__":
    # Replace with your actual CSV file path
    your_csv_path = "your_file.csv"
    
    # Convert CSV to dictionary
    utterance_dict = csv_to_utterance_dict(your_csv_path)
    
    # Print sample output (check utterance1 for sample_id 0)
    print("Utterance1 content for sample_id 0:")
    print(utterance_dict[0]['utterance1'])
    print("\nList of all sample IDs:", list(utterance_dict.keys()))