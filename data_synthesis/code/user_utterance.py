"""
Step 3: Synthesize user utterance
"""

import os
import json
from tqdm import tqdm
import pandas as pd
from copy import copy, deepcopy

import sys
sys.path.append('compass/data_synthesis/code')
from llm_generation import LLM_Sequential_Generation

class UntteranceGeneration(LLM_Sequential_Generation):
    def __init__(self, users_dict, scenario_dict,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scenario_dict = scenario_dict
        self.users_dict = users_dict
        self.data_synthesis_root = 'compass/data_synthesis'


    def get_prompt(self, sample_id):
        prompt_template_dir = os.path.join(self.data_synthesis_root, 'prompt', 'user_utterance' )
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            system_prompt = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            user_prompt_template = ''.join(lines)
        
        user_prompt = user_prompt_template.replace('<persona_data>', json.dumps(self.users_dict[sample_id], indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace('<scenario_data>', json.dumps(self.scenario_dict[sample_id], indent=4, ensure_ascii=False))

        return system_prompt, user_prompt


    def check_generation(self, sample_id, raw_generation):
        generation = self.generation_postprocess(raw_generation)
        try:
            generation_dict = json.loads(generation)
            if list(generation_dict.keys()) != [
    "utterance1", "utterance2", "utterance3", "utterance4", "utterance5",
    "utterance6", "utterance7", "utterance8", "utterance9", "utterance10",
    "utterance11", "utterance12", "utterance13", "utterance14", "utterance15"
]:
                return False, "wrong key"
            for k in list(generation_dict.keys()):
                if generation_dict[k].strip() == "" or generation_dict[k] == "...":
                    return False, "empty requirement"
        except:
            return False, "JSON format error"
        return True, None


def generate_utterance_main(users_dict_dir,scenario_dict_dir,model="gpt-4.1"):
    model_name = model
    with open(users_dict_dir, "r", encoding="utf-8") as f:
            users_dict = json.load(f)
    with open(scenario_dict_dir, "r", encoding="utf-8") as f:
            scenario_dict = json.load(f)
    data_synthesis_save_root = 'compass/data_synthesis/data'
    save_dir = os.path.join(data_synthesis_save_root, 'debug', 'generate_utterance')
    utterance_generator = UntteranceGeneration(users_dict, scenario_dict,model_name=model_name, save_dir=save_dir,sample_id_list=list(users_dict.keys()))

    # # check the prompt
    # system_prompt, user_prompt = personality_generator.get_prompt(sample_id='0000')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    # Generate output
    utterance_generator.sequential_generate()

if __name__ == "__main__":
     users_dict_dir="compass/data_synthesis/data/persona/test1.json"
     scenario_dict_dir="compass/data_synthesis/data/scenarios/scenarios.json"
     generate_utterance_main(users_dict_dir,scenario_dict_dir)

