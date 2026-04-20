prompt:
````
【System Prompt】
You are an excellent user data synthesizer. You can synthesize personalized user information for me according to my instructions.
【User Prompt】
You will be given a user's basic information. Based on this, generate background descriptions for the user in theses specified aspects below:
- Hobbies: Must include 1-2 specific interest scenarios (e.g., "Every Saturday afternoon, he goes to the café near his home to write short stories" or "He participates in urban peripheral hiking activities once a month, and his usual equipment is a pair of hiking shoes from [a specific brand]"). Avoid general descriptions and reflect the regularity or detailed characteristics of the interests;
- Health Status: Must clearly state the physical condition (e.g., "He has no underlying diseases, undergoes a comprehensive medical check-up once a year, and the report shows that his blood lipid index is within the normal range" or "Due to long-term sedentary work, he occasionally suffers from cervical soreness and does cervical rehabilitation exercises three times a week to relieve it"). You may supplement relevant health details in combination with occupational characteristics;
- Family Environment: Must explain the family structure and core interaction mode (e.g., "He lives with his parents. His father is a retired teacher and his mother is a community volunteer. Every Sunday evening, the whole family cooks dinner together and watches documentaries" or "He lives alone and keeps a 3-year-old orange cat. He has video calls with his college roommates 2-3 times a month and goes back to his hometown to spend the Spring Festival with his parents every year");
- Living Habits: Must cover specific content such as work-rest schedule, diet, and daily behaviors (e.g., "He gets up at 7 a.m. every day, drinks a glass of warm water first, then does 15 minutes of yoga. For breakfast, he often eats whole-wheat bread with fried eggs and milk, and goes to bed before 11 p.m. with very few late nights" or "On workdays, he usually has lunch at a light meal restaurant downstairs from his company, and is used to cooking dinner by himself. Every Wednesday and Friday evening, he goes to the gym to do 40 minutes of spinning");
- Growth Experience: Must include 1-2 key stages or events (e.g., "When she was in primary school, she joined the school choir and once represented the school in a municipal chorus competition and won the second prize. This experience makes her still like singing in her spare time" or "She majored in computer science in college. In her junior year, she participated in a campus programming competition. Although she didn't win an award, she accumulated practical experience, which laid a foundation for her to work as a developer in an Internet company after graduation" or "When she was in high school, she transferred to another school due to her parents' job transfer. The head teacher of the new school paid great attention to her and often talked with her to help her quickly adapt to the new environment. This experience makes her willing to take the initiative to help people around her who need to adapt to a new environment later").

# Task Input
## User Information
"""
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
    'personality': {
        'Openness': 'high', 
        'Conscientiousness': 'low', 
        'Extraversion': 'low', 
        'Agreeableness': 'medium', 
        'Neuroticism': 'low'
    }, 
    'personality_description': 'The user is a creative and open-minded individual, who prefers solitude and spontaneity over structure, while maintaining a balanced and calm demeanor in interactions.'
}
"""

# Specific Requirements and Output Format
## Specific Requirements
1. Each dimension must be fully described in 2-3 sentences of natural language.
2. The language must be objective and affirmative, and vague words such as "may", "uncertain", "seem", and "tend to" are strictly prohibited.
3. The content must be specific in details, conform to the logic of the character's identity, and avoid general expressions.
4. Follow the output format below. Do not include anything else or add extra comments in the JSON.

## Output Format (json)
"""
{
    "Hobbies": "...",
    "Health Status": "...",
    "Family Environment": "...",
    "Living Habits": "...",
    "Growth Experience": "..."
}
"""

Now, please provide the output according to the above requirements and output format.
````

Output:
````
{
    "Hobbies": "Burnice is passionate about independent cinema and maintains a personal blog where she writes detailed reflections on films she watches every Tuesday evening at the local arthouse theater in Cruzchester. Additionally, she enjoys abstract painting and has a small studio set up in her attic, where she paints on Sunday mornings while listening to classical music playlists she curates herself.",
    "Health Status": "She is in generally good physical condition without any chronic illnesses and undergoes a comprehensive medical check-up every autumn; her most recent results showed stable blood pressure and normal cholesterol levels. Due to long hours spent reading and writing, she experiences occasional eye strain, which she addresses by using blue-light filtering glasses and limiting screen time in the evenings.",
    "Family Environment": "Burnice lives alone in a two-story townhouse in Cruzchester and has a close emotional connection with her younger brother, who resides in Basel. They have a fixed tradition of connecting via video call every Thursday evening, during which they share updates about their week and recommend new books or podcasts to each other.",
    "Living Habits": "She wakes up at 6:45 a.m. daily and begins her day with 20 minutes of meditation while brewing a pot of green tea. Her meals are mostly plant-based, and on weekdays she prepares simple dishes like quinoa salad or roasted vegetables for dinner. She avoids late nights, typically going to bed around 10:30 p.m., and spends an hour before sleep reading literary fiction by European authors.",
    "Growth Experience": "In her late 30s, Burnice took a year-long sabbatical to travel alone across Central and Eastern Europe, documenting small cultural stories that later inspired her to pursue a master's degree in cultural studies. During her undergraduate years, she participated in a cross-border academic exchange program with a university in Austria, which contributed significantly to her lifelong interest in comparative storytelling traditions."
}
````