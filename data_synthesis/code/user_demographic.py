"""
Step 1: Synthesize user persona
"""
import os
import json
from llm_generation import LLM_Individual_Generation
from generate_persona.generate_age import generate_random_age
from generate_persona.generate_gender import generate_random_gender
from generate_persona.generate_address_name import generate_american_info
from generate_persona.generate_edu_by_age import get_education_by_age
from generate_persona.generate_career import get_random_occupation
from generate_persona.generate_income import IncomeGeneration
from generate_persona.generate_Big_Five_personality import generate_personality_traits
from generate_persona.generate_personality import PersonalityGeneration
from generate_persona.generate_background_batch import BackgroundGeneration
from generate_persona.generate_tool_perference import ToolPerferenceGeneration

def generate_single_persona_demographic():
    """Generate basic demographic info for a single user (excluding background & tool preferences)"""
    persona = {"demographic": {}}
    # Generate basic demographic information
    persona["demographic"]["age"] = generate_random_age()
    persona["demographic"]["gender"] = generate_random_gender()
    success, firstname, lastname, address, phone = generate_american_info(gender=persona["demographic"]["gender"])
    persona["demographic"]["firstname"] = firstname
    persona["demographic"]["lastname"] = lastname
    persona["demographic"]["address"] = address
    persona["demographic"]["phone"] = phone
    persona["demographic"]["education"] = get_education_by_age(persona["demographic"]["age"])
    
    # Generate occupation
    career_path = 'compass/data_synthesis/code/generate_persona/ISCO-08 -88 EN Index.xlsx'
    success1, persona["demographic"]["career"] = get_random_occupation(persona["demographic"]["education"], file_path=career_path)
    
    # Generate income
    users_dict = {'temp': persona}
    save_dir = os.path.join('compass/data_synthesis/data', 'debug', 'generate_income')
    income_generator = IncomeGeneration(users_dict, model_name="gpt-4.1", save_dir=save_dir)
    success2, user_income = income_generator.generate(sample_id='temp')
    persona["demographic"]["income"] = json.loads(user_income)['Income']
    
    # Generate Big Five personality and description
    persona["Big_Five_personality_traits"] = generate_personality_traits()
    users_dict['temp'] = persona
    save_dir = os.path.join('compass/data_synthesis/data', 'debug', 'generate_personality')
    personality_generator = PersonalityGeneration(users_dict, model_name="gpt-4.1", save_dir=save_dir)
    success3, user_personality = personality_generator.generate(sample_id='temp')
    persona["personality_description"] = json.loads(user_personality)['Personality']
    
    return persona

def generate_persona_batch(amount=5, model_name="gpt-4.1"):
    """Generate user personas in batch (call LLM for background every 5 users)"""
    success = True
    batch_persona = {}
    
    # Step 1: Generate basic info for all users
    users_dict = {}
    for i in range(amount):
        user_id = f"{i}"
        single_persona = generate_single_persona_demographic()
        # Extract info needed for background generation
        users_dict[user_id] = {
            'demographic': single_persona["demographic"],
            'personality': single_persona["Big_Five_personality_traits"],
            'personality_description': single_persona["personality_description"]
        }
        batch_persona[user_id] = single_persona
        print(f"Basic info generated for user {i}")
    print("demographic complete")
    
    # Step 2: Generate backgrounds in groups of 5 users
    data_synthesis_save_root = 'compass/data_synthesis/data'
    save_dir = os.path.join(data_synthesis_save_root, 'debug', 'background')
    
    # Split user IDs into groups of 5
    user_ids = list(users_dict.keys())
    batch_size = 5
    samples_path = "compass/data_synthesis/prompt/user_persona/generate_background/sample.json"
    with open(samples_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    samples = samples["samples"]
    num_samples = len(samples)
    
    for i in range(0, len(user_ids), batch_size):
        # Get user IDs and data for current batch
        sample = samples[(i // batch_size) % num_samples]
        sample = str(sample)
        batch_user_ids = user_ids[i:i + batch_size]
        batch_users_dict = {user_id: users_dict[user_id] for user_id in batch_user_ids}
        
        # Call LLM to generate background for current batch
        background_generator = BackgroundGeneration(batch_users_dict, model_name=model_name, save_dir=save_dir, sample=sample)
        success1, batch_background = background_generator.generate_batch()
        if not success1:
            return False, f"Background generation failed for batch {i//batch_size + 1}: {batch_background}"
        
        # Assign generated backgrounds to users
        for user_id in batch_user_ids:
            batch_persona[user_id]["background"] = batch_background[user_id]
        print(f"Background generated for users {i} to {i+5}")
    
    # Step 3: Generate tool preferences for each user
    for user_id in batch_persona.keys():
        users_dict_single = {'0000': batch_persona[user_id]}
        save_dir_tool = os.path.join(data_synthesis_save_root, 'debug', 'generate_tool_perference')
        tool_generator = ToolPerferenceGeneration(users_dict_single, model_name=model_name, save_dir=save_dir_tool)
        success2, tool_preference = tool_generator.generate(sample_id='0000')
        if not success2:
            return False, f"Tool preference generation failed for user {user_id}: {tool_preference}"
        batch_persona[user_id]["tool_preferences"] = json.loads(tool_preference)
        print(f"Preferences generated for user {user_id}")
    
    return True, batch_persona

def generate_mul_persona(amount=5, model="gpt-4.1"):
    """Generate multiple user personas (backward-compatible interface)"""
    return generate_persona_batch(amount=amount, model_name=model)

def generate_persona_main(amount=5, save_path=None, model="gpt-4.1"):
    """Main function: Generate and save user personas"""
    if save_path is None:
        save_path = "compass/data_synthesis/data/persona/test3.json"
    
    success, user_dict = generate_mul_persona(amount=amount, model=model)
    if success:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(user_dict, f, ensure_ascii=False, indent=4)
        print(f"Successfully generated {amount} personas, saved to {save_path}")
    else:
        print(f"Generation failed: {user_dict}")

if __name__ == "__main__":
    save_path = "compass/data_synthesis/data/persona/batch_10_personas.json"
    model = "gpt-4.1"
    generate_persona_main(amount=10, save_path=save_path, model=model)