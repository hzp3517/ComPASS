"""
Step 2: Synthesize user scenario
"""
from APIclient import onechatAPIclient
import json
from tqdm import tqdm
import random
from EDretriever import retriever
import os
import logging


class ScenarioGenerator:
    def __init__(self, data, output_file: str, amount=15):
        """输入profile 输出scenario"""
        self.client = onechatAPIclient(model="gpt-4.1")
        self.output_file = output_file
        self.user_profiles = data
        self.amount = amount


    def load_user_profile(self):
        """with open(self.profile_path, "r", encoding="utf-8") as f:
            return json.load(f)"""
        with open(self.profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [data] 
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("Profile file must be JSON dict or list of dicts")
        

    def build_prompt(self, existing_scenarios, step, profile, mainline, prompt_file_path="compass/data_synthesis/prompt/user_scenario/user_prompt.txt",system_prompt_file_path="compass/data_synthesis/prompt/user_scenario/system_prompt.txt"):
        base_info = json.dumps(profile, ensure_ascii=False, indent=2)
        prev_scenarios = json.dumps(existing_scenarios, ensure_ascii=False, indent=2) if existing_scenarios else "none"
        emotions = [
                    "afraid", "angry", "annoyed", "anticipating", "anxious", "apprehensive", "ashamed",
                    "caring", "confident", "content", "devastated", "disappointed",
                    "embarrassed", "excited", "faithful", "furious", "grateful", "guilty", "hopeful",
                    "impressed", "jealous", "joyful", "lonely", "nostalgic", "prepared", "proud",
                    "trusting", "sad", "sentimental", "surprised", "terrified"
                ]
        emotion = random.choices(emotions, k=step) #list
        EDref = []
        for i in range(step):
            ed = retriever(emotion[i], mainline)
            EDref.append(ed)
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        prompt = prompt.format(
            base_info=base_info,
            prev_scenarios=prev_scenarios,
            step=step,
            emotion=emotion,
            EDref=EDref
        )

        with open(system_prompt_file_path, 'r', encoding='utf-8') as f:
            sys_prompt = f.read().strip()
        return sys_prompt, prompt, emotion

    def generateMainline(self, profile):
        base_info = json.dumps(profile, ensure_ascii=False, indent=2)
        prompt = f"""
            You are a scenario generation assistant.You should generate a person's storyline in a year,a person should experience both joyful thing and sad incidents 
            describe his rise and falls in life. Below is the user profile:
            {base_info}
            requirement:
            - should include various aspects of life, and experiences should be as rich as possible
            - please try to consider all aspects of experience
            """
        system_prompt = "You are an assistant that helps construct coherent and emotionally relevant life scenarios based on user profiles."
        response = self.client.apicall(system_prompt=system_prompt, user_prompt=prompt)
        print("MAINLINE:\n")
        print(response)
        print("--------------")
        return response

        
    def generate(self, profile, total=15, step=5, mainline=None, repeat=0):
        scenarios = []
        with tqdm(total=total, desc="Generating scenarios") as pbar:
            while len(scenarios) < total:
                system_prompt, prompt, emotion= self.build_prompt(scenarios, step, profile, mainline)
                response = self.client.apicall(system_prompt=system_prompt, user_prompt=prompt) 
                print(f"RESPONSE:{response}")
                moveon = 0
                while repeat<10:
                    
                    try:
                        new_scenarios = json.loads(response)
                        #print(f"st:{new_scenarios}")
                        if isinstance(new_scenarios, list):
                            for i, sc in enumerate(new_scenarios):
                                if isinstance(sc, dict):
                                    sc["emotion"] = emotion[i]
                                else:
                                    new_scenarios[i] = {"scenario": sc, "emotion": emotion[i]}
                        else:
                            new_scenarios = [{"scenario": new_scenarios, "emotion": emotion[0]}]
                        scenarios.extend(new_scenarios)
                        scenarios = scenarios[:total]
                        pbar.n = len(scenarios) 
                        pbar.refresh()  
                        moveon = 1  
                        break 
                    except Exception as e:
                        repeat += 1
                        print("ERROR while extracting...", e)
                        response = self.client.apicall(system_prompt=system_prompt, user_prompt=prompt)
                if moveon == 1:
                    continue
                logging.basicConfig(
                    filename="compass/data_synthesis/data/debug/generate_scenario/err.log",
                    level=logging.INFO,      
                    format="%(asctime)s [%(levelname)s] %(message)s"
                )
                logging.info(f"Start generating scenarios{prompt}")
                logging.error("Failed after 10 retries")               
        return scenarios  

    def generate_all(self):
        output_data = {}
        for key in tqdm(self.user_profiles, desc="Generating scenarios for all users"):
            profile=self.user_profiles[key]
            name = profile.get("demographic", {}).get("firstname") or profile.get("demographic", {}).get("name", "Unknown")
            #print(profile)
            #print(name)
            mainline = self.generateMainline(profile)
            scenarios = self.generate(profile, total=self.amount, step=3, mainline=mainline,repeat=0)
            output_data[key] = {
                "name": name,
                "scenarios": scenarios
            }
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Saved all scenarios to {self.output_file}")

def generate_scenario_main(file_dir,outdata_dir,amount):

    with open(file_dir, "r", encoding="utf-8") as f:
            data = json.load(f)
    generator = ScenarioGenerator(data=data, output_file=outdata_dir, amount=amount)
    generator.generate_all()
    print(f"YYY SUCCESS!saved in {outdata_dir}")
if __name__ == "__main__":
    persona_path="compass/data_synthesis/data/persona/test.json"
    output_path="compass/data_synthesis/data/scenarios/scenarios.json"
    generate_scenario_main(persona_path,output_path)

        

