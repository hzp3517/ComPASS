import os
import json
from tqdm import tqdm
import pandas as pd
from copy import copy, deepcopy
import sys
sys.path.append('compass/data_synthesis/code')
from llm_generation import LLM_Individual_Generation

class BackgroundGeneration(LLM_Individual_Generation):
    def __init__(self, users_dict,sample, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_dict = users_dict  
        self.data_synthesis_root = 'compass/data_synthesis'
        self.sample=sample

    def get_batch_prompt(self):
        prompt_template_dir = os.path.join(self.data_synthesis_root, 'prompt', 'user_persona', 'generate_background')
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            system_prompt = ''.join(f.readlines())
        
        with open(os.path.join(prompt_template_dir, 'user_prompt_batch.txt'), 'r', encoding="utf-8") as f:
            user_prompt_template = ''.join(f.readlines())
        
        users_info_str = ""
        for idx, (user_id, user_info) in enumerate(self.users_dict.items(), 1):
            users_info_str += f"### User {idx} (ID: {user_id})\n{json.dumps(user_info, indent=2, ensure_ascii=False)}\n\n"
        user_prompt = user_prompt_template.replace('<users_info>', users_info_str.strip())
        user_prompt = user_prompt.replace('<sample>', self.sample)
        
        return system_prompt, user_prompt

    def check_generation(self, generation_dict):
        required_keys = ['Hobbies', 'Health Status', 'Family Environment', 'Living Habits', 'Growth Experience']
        if list(generation_dict.keys()) != required_keys:
            return False, "wrong key"
        for k in required_keys:
            if not generation_dict[k].strip() or generation_dict[k] == "...":
                return False, "empty requirement"
        return True, None

    def generate_batch(self):
        system_prompt, user_prompt = self.get_batch_prompt()
        raw_generation = self.generate_raw(system_prompt, user_prompt) 
        generation = self.generation_postprocess(raw_generation)
        
        try:
            batch_background = json.loads(generation)
            result = {}
            for user_id, background in zip(self.users_dict.keys(), batch_background):
                valid, msg = self.check_generation(background)
                if not valid:
                    return False, f"User {user_id} background error: {msg}"
                result[user_id] = background
            return True, result
        except Exception as e:
            return False, f"Batch JSON format error: {str(e)}"

if __name__ == "__main__":
    model_name = 'gpt-4.1'
    users_dict = {}
    for i in range(5):
        user_id = f"000{i}"
        users_dict[user_id] = {
            'demographic': {
                'age': 59 + i,
                'gender': 'female' if i % 2 == 0 else 'male',
                'firstname': f'Name{i}',
                'lastname': 'Crooks',
                'country': 'Liechtenstein',
                'city': 'Cruzchester',
                'zipcode': '52257-2155',
                'street': '562 Cristian Light',
                'education': "Master's degree (in progress)" if i < 3 else "Bachelor's degree"
            },
            'personality': {
                'Openness': 'high' if i % 2 == 0 else 'medium',
                'Conscientiousness': 'low',
                'Extraversion': 'low',
                'Agreeableness': 'medium',
                'Neuroticism': 'low'
            },
            'personality_description': f'The user is creative and open-minded, prefers solitude and spontaneity (User {i}).'
        }
    
    data_synthesis_save_root = 'compass/data_synthesis/data'
    save_dir = os.path.join(data_synthesis_save_root, 'debug', 'background')
    background_generator = BackgroundGeneration(users_dict, model_name=model_name, save_dir=save_dir)
    
    success, batch_user_background = background_generator.generate_batch()
    if success:
        for user_id, background in batch_user_background.items():
            print(f"【User {user_id} Background】\n{json.dumps(background, indent=2, ensure_ascii=False)}\n")
    else:
        print(f"Generation failed: {batch_user_background}")