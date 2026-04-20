import random
from typing import Union

def generate_random_age(
    age_distribution: dict = None, 
    seed: Union[int, None] = None
) -> int:
    if age_distribution is None:
        age_distribution = {
            "15-24": 22.62,
            "25-54": 55.83,
            "55-64": 11.09,
            "65+": 10.46
        }
    

    total_weight = sum(age_distribution.values())
    weights = [val / total_weight for val in age_distribution.values()] 
    

    age_ranges = [   
        (15, 24),   
        (25, 54),   
        (55, 64),   
        (65, 85)  
    ]
    

    if seed is not None:
        random.seed(seed)
    

    selected_range = random.choices(age_ranges, weights=weights, k=1)[0]
    

    random_age = random.randint(selected_range[0], selected_range[1])
    
    return random_age


if __name__ == "__main__":

    print("Generate 10 random ages:")
    for i in range(10):
        print(f"Age {i+1}: {generate_random_age()}")
    

    print("\nAge generated with fixed seed (seed=456):")
    print(generate_random_age(seed=456))
    print(generate_random_age(seed=456))
    
    print("\nAge distribution statistics from 10,000 samples:")
    age_counts = {range_name: 0 for range_name in ["0-14", "15-24", "25-54", "55-64", "65+"]}
    total_samples = 10000
    
    for _ in range(total_samples):
        age = generate_random_age()
        if 0 <= age <= 14:
            age_counts["0-14"] += 1
        elif 15 <= age <= 24:
            age_counts["15-24"] += 1
        elif 25 <= age <= 54:
            age_counts["25-54"] += 1
        elif 55 <= age <= 64:
            age_counts["55-64"] += 1
        else:
            age_counts["65+"] += 1
    
    for range_name, count in age_counts.items():
        percentage = (count / total_samples) * 100
        print(f"{range_name} years old: {count} people, percentage: {percentage:.2f}%")