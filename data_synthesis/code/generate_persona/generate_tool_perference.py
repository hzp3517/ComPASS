import os
import json
from tqdm import tqdm
import pandas as pd
from copy import copy, deepcopy

import sys
sys.path.append('compass/data_synthesis/code')
from llm_generation import LLM_Individual_Generation

class ToolPerferenceGeneration(LLM_Individual_Generation):
    def __init__(self, users_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_dict = users_dict
        self.data_synthesis_root = 'compass/data_synthesis'


    def get_prompt(self, sample_id):
        prompt_template_dir = os.path.join(self.data_synthesis_root, 'prompt', 'user_persona', 'generate_tool_perference')
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            system_prompt = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            user_prompt_template = ''.join(lines)
        
        user_prompt = user_prompt_template.replace('<user_info>', json.dumps(self.users_dict[sample_id], indent=4, ensure_ascii=False))

        return system_prompt, user_prompt


    def check_generation(self, sample_id, raw_generation):
        generation = self.generation_postprocess(raw_generation)
        try:
            generation_dict = json.loads(generation)
            if list(generation_dict.keys()) != ["Work/Study_scenario","Health_scenario","Family_scenario","Social_scenario","Leisure_scenario"]:
                return False, "wrong key"
            for k in list(generation_dict.keys()):
                if generation_dict[k].strip() == "" or generation_dict[k] == "...":
                    return False, "empty requirement"
        except:
            return False, "JSON format error"
        return True, None
    def generation_postprocess(self, raw_generation_str):
        if '```json' in raw_generation_str:
            raw_generation_str = raw_generation_str.split('```json')[-1]
            if '```' in raw_generation_str:
                raw_generation_str = raw_generation_str.split('```')[0]
        raw_generation_str = raw_generation_str.replace('```', '').strip()
        generation_str = raw_generation_str.replace('\"\"\"', '').strip()
        generation_str = generation_str.replace('""','"').strip()
        return generation_str
    

if __name__ == "__main__":
    model_name = 'gpt-4.1'
    ### -------------------------

    # Input user information
    user_info = {
    "demographic": {
        "age": 31,
        "gender": "female",
        "firstname": "Kirsten",
        "lastname": "Toy",
        "country": "Guernsey",
        "city": "Reynoldsfort",
        "zipcode": "79700",
        "street": "449 Murray Causeway",
        "education": "Vocational high school graduate",
        "career": "Worker, insulation",
        "tool_perference": "35,000-80,000 USD per year"
    },
    "Big_Five_tool_perference_traits": {
        "Openness": "high",
        "Conscientiousness": "medium",
        "Extraversion": "high",
        "Agreeableness": "medium",
        "Neuroticism": "high"
    },
    "tool_perference_description": "This person is sociable, energetic, and imaginative, embraces new experiences, keeps a moderate level of organization, is generally cooperative yet candid, and is emotionally sensitive and prone to stress.",
    "background": {
        "Hobbies": "Every Saturday at 7 a.m., she meets a local women’s open-water group at Vazon Bay for a 20-minute sea swim, then warms up with a hot chocolate at the Richmond kiosk while logging water temps in a small notebook. Every other Wednesday evening, she attends a pottery wheel class at The Kiln in St Peter Port, where she throws small bowls and tests celadon and shino glazes that she documents in a dedicated sketchbook.",
        "Health status": "She has no chronic illnesses and completes an annual check-up at Princess Elizabeth Hospital; her latest report lists blood pressure at 112/72, resting heart rate at 64, and labs within normal ranges. Due to handling fiberglass and foam boards, she occasionally experiences forearm and neck tightness, so she wears an FFP3 mask, nitrile gloves, and goggles on site and does 20 minutes of shoulder and hip mobility exercises three times a week to prevent strain.",
        "Family environment": "She rents a two-bedroom flat at 449 Murray Causeway in Reynoldsfort with her close friend Aimee, a pediatric nurse; they keep a shared chore rota and cook seafood pasta together every Tuesday while catching up on each other’s week. Her parents live in St Sampson, and the family gathers for a roast lunch on the first Sunday of each month, after which she takes her 6-year-old niece to the playground so her sister and brother-in-law can have an hour to themselves.",
        "Living habits": "On workdays she wakes at 6:00 a.m., drinks a glass of water, eats porridge with a banana and black tea, and bikes 15 minutes to the site to start at 7:30 a.m.; lunch is a chicken-and-avocado wrap with an apple, and she refills a 1.5-liter bottle twice. After work she showers immediately to remove fibers, stretches on Monday, Wednesday, and Friday for 20 minutes, avoids caffeine after 2:00 p.m., shops at the Town Market on Saturday mornings, and aims to be in bed by 10:45 p.m.",
        "Growth experience": "In vocational high school she chose the building services track and completed a two-year apprenticeship with Sarnian Insulation; in her final year, her team won first place in the Guernsey Skills Challenge for a retrofitting project, which led directly to her current role. At 19, she helped her family repair her grandparents’ storm-damaged cottage in Torteval over three weekends, coordinating materials and schedules, and the successful rebuild cemented her confidence in construction work."
    }
}
    users_dict = {'0000': user_info}

    data_synthesis_save_root = 'compass/data_synthesis/data'
    save_dir = os.path.join(data_synthesis_save_root, 'debug', 'generate_tool_perference') 

    tool_perference_generator = ToolPerferenceGeneration(users_dict, model_name=model_name, save_dir=save_dir)
    
    # # check the prompt
    # system_prompt, user_prompt = tool_perference_generator.get_prompt(sample_id='0000')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    # Generate output
    success, user_tool_perference = tool_perference_generator.generate(sample_id='0000')
    print(user_tool_perference)   
    