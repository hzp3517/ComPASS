prompt:
````
【System Prompt】
Supplement content for the following five dimensions of the user profile. You may fabricate non-existent content to enrich the character image:
    - Hobbies:Must include 1-2 specific interest scenarios (e.g., "Every Saturday afternoon, he goes to the café near his home to write short stories" or "He participates in urban peripheral hiking activities once a month, and his usual equipment is a pair of hiking shoes from [a specific brand]"). Avoid general descriptions and reflect the regularity or detailed characteristics of the interests;
    - Health status:Must clearly state the physical condition (e.g., "He has no underlying diseases, undergoes a comprehensive medical check-up once a year, and the report shows that his blood lipid index is within the normal range" or "Due to long-term sedentary work, he occasionally suffers from cervical soreness and does cervical rehabilitation exercises three times a week to relieve it"). You may supplement relevant health details in combination with occupational characteristics;
    - Family environment:Must explain the family structure and core interaction mode (e.g., "He lives with his parents. His father is a retired teacher and his mother is a community volunteer. Every Sunday evening, the whole family cooks dinner together and watches documentaries" or "He lives alone and keeps a 3-year-old orange cat. He has video calls with his college roommates 2-3 times a month and goes back to his hometown to spend the Spring Festival with his parents every year");
    - Living habits: Must cover specific content such as work-rest schedule, diet, and daily behaviors (e.g., "He gets up at 7 a.m. every day, drinks a glass of warm water first, then does 15 minutes of yoga. For breakfast, he often eats whole-wheat bread with fried eggs and milk, and goes to bed before 11 p.m. with very few late nights" or "On workdays, he usually has lunch at a light meal restaurant downstairs from his company, and is used to cooking dinner by himself. Every Wednesday and Friday evening, he goes to the gym to do 40 minutes of spinning");
    - Growth experience:Must include 1-2 key stages or events (e.g., "When she was in primary school, she joined the school choir and once represented the school in a municipal chorus competition and won the second prize. This experience makes her still like singing in her spare time" or "She majored in computer science in college. In her junior year, she participated in a campus programming competition. Although she didn't win an award, she accumulated practical experience, which laid a foundation for her to work as a developer in an Internet company after graduation" or "When she was in high school, she transferred to another school due to her parents' job transfer. The head teacher of the new school paid great attention to her and often talked with her to help her quickly adapt to the new environment. This experience makes her willing to take the initiative to help people around her who need to adapt to a new environment later").
    Each dimension must be fully described in 2-3 sentences of natural language. The language must be objective and affirmative, and vague words such as "may", "uncertain", "seem", and "tend to" are strictly prohibited; the supplementary content must be specific in details, conform to the logic of the character's identity, and avoid general expressions.
    Ensure the answer format is clear, using the specified five sections, each starting with "- Name:".
【User Prompt】
{
    'demographic': {
        'age': 59, 
        'gender': 'female', 
        'firstname': 'Burnice', 
        'lastname': 'Crooks', 
        'country': 'Liechtenstein', 
        'city': 'Cruzchester', 
        'zipcode': '52257-2155', 
        'street': '562 Cristian Light', 
        'education': "Master's degree (in progress)"
    }, 
    'ocean': {
        'Openness': 'high', 
        'Conscientiousness': 'low', 
        'Extraversion': 'low', 
        'Agreeableness': 'medium', 
        'Neuroticism': 'low'
    }, 
    'personality_description': 'The user is a creative and open-minded individual, who prefers solitude and spontaneity over structure, while maintaining a balanced and calm demeanor in interactions.'
}
````