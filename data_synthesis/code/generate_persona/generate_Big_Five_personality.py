import random

def generate_personality_traits():

    personalities = [
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism"
    ]
    
    while True:

        results = []
        for _ in range(5):
            result = random.choice(["high", "low", "medium"])
            results.append(result)
        

        if results.count("medium") < 3:

            return dict(zip(personalities, results))


    