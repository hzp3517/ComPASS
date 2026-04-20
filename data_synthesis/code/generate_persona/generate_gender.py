import random
from typing import Union, Literal

def generate_random_gender(
    gender_ratio: dict = None,
    seed: Union[int, None] = None,
    output_type: Literal['string', 'char', 'code'] = 'string'
) -> Union[str, int]:
    """
    Generates a random gender. Default gender ratio is 1:1 for male:female.
    Custom ratio and output format can be specified.
    
    Parameters:
        gender_ratio: Dictionary with genders as keys and their weights as values.
                      Defaults to 1:1 male:female ratio if not provided.
        seed: Integer random seed (for reproducibility, set to None for random results).
        output_type: Output format, options:
                    - 'string': Returns "male" or "female"
                    - 'char': Returns the first letter of the gender
                    - 'code': Returns numeric code (1 for male, 0 for female)
        
    Returns:
        Gender representation based on the specified output_type.
    """
    # 1. Set default gender ratio (1:1 male:female)
    if gender_ratio is None:
        gender_ratio = {
            "male": 1,
            "female": 1
        }
    
    # 2. Extract gender list and corresponding weights
    genders = list(gender_ratio.keys())
    weights = list(gender_ratio.values())
    
    # 3. Set random seed (optional, for reproducibility)
    if seed is not None:
        random.seed(seed)
    
    # 4. Randomly select a gender based on weights
    selected_gender = random.choices(genders, weights=weights, k=1)[0]
    
    # 5. Return result in the specified format
    if output_type == 'string':
        return selected_gender
    elif output_type == 'char':
        return selected_gender[0]
    elif output_type == 'code':
        return 1 if selected_gender == "male" else 0
    else:
        raise ValueError("output_type must be 'string', 'char', or 'code'")


# Test 1: Verify 1:1 ratio (10,000 samples)
if __name__ == "__main__":
    print("Gender ratio statistics from 10,000 samples (1:1):")
    gender_counts = {"male": 0, "female": 0}
    total_samples = 10000
    
    for _ in range(total_samples):
        gender = generate_random_gender()
        gender_counts[gender] += 1
    
    male_percent = (gender_counts["male"] / total_samples) * 100
    female_percent = (gender_counts["female"] / total_samples) * 100
    ratio = gender_counts["male"] / gender_counts["female"]
    
    print(f"male: {gender_counts['male']}, {male_percent:.2f}%")
    print(f"female: {gender_counts['female']}, {female_percent:.2f}%")
    print(f"Gender ratio (male:female): 1:{ratio:.2f}")