from encodings import utf_8
import json
from pyexpat import model
from APIclient import onechatAPIclient, api_key
import base64
import os
import re
from llm_generation import LLM_Individual_Generation
import copy

class ToolCheckGeneration(LLM_Individual_Generation):
    def __init__(self, users_dict, user_utterance,tool_generation,user_scenario,type,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.user_utterance=user_utterance
        self.tool_generation=tool_generation
        self.user_scenario=user_scenario
        self.type=type
        self.data_synthesis_root = 'compass/data_synthesis'
        persona=copy.deepcopy(users_dict["0000"])
        #print(persona)
        persona.pop('tool_preferences')
        self.users_dict =persona


    def get_prompt(self, sample_id):

        prompt_template_dir = os.path.join(self.data_synthesis_root, 'prompt', 'tool_check')
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            system_prompt = ''.join(lines)
        if self.type=='pos':
            with open(os.path.join(prompt_template_dir, 'user_prompt_text_pos.txt'), 'r', encoding="utf-8") as f:
                lines = f.readlines()
                user_prompt_template = ''.join(lines)
        elif self.type=='neg':
            with open(os.path.join(prompt_template_dir, 'user_prompt_text_pos.txt'), 'r', encoding="utf-8") as f:
                lines = f.readlines()
                user_prompt_template = ''.join(lines)
        user_prompt = user_prompt_template.replace('<persona_data>', json.dumps(self.users_dict, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace('<scenario_data>', json.dumps(self.user_scenario, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace('<utterance_data>', json.dumps(self.user_utterance, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace('<tool_call>', json.dumps(self.tool_generation["generated_feedback"], indent=4, ensure_ascii=False))
        tool = self.tool_generation['tool_call']['tool_folder_name']
        if tool == 'sticker_respond':
            image_base64=""
            pattern = r"/[\w\/\.]+?\.(jpg|png|gif)"
            match = re.search(pattern, self.tool_generation["generated_feedback"])
            if match:
                print("match")
                image_path = match.group(0)
                if os.path.exists(image_path):
                    with open(image_path, "rb") as f:
                        image_base64 = base64.b64encode(f.read()).decode("utf-8")
            if  image_base64:
                print("exist url")
                user_prompt=[
            {
            "type": "text",
            "text": f"{user_prompt}"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpg;base64,{image_base64}"
            }
            }
                            ]
                #print(user_prompt)
        return system_prompt, user_prompt


    def check_generation(self, sample_id, raw_generation):
        generation = self.generation_postprocess(raw_generation)
        try:
            generation_dict = json.loads(generation)
            if list(generation_dict.keys()) != ['score','reasons']:
                return False, "wrong key"
            for k in list(generation_dict.keys()):
                if generation_dict[k].strip() == "" or generation_dict[k] == "...":
                    return False, "empty requirement"
        except:
            return False, "JSON format error"
        return True, None



if __name__ == "__main__":
    pass