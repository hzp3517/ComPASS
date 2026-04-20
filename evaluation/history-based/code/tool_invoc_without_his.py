"""
Tool Call Generation for Evaluation - without history version
"""
import json
import os
import importlib.util
from typing import List, Dict
import sys
import random
import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add custom module path (update this to your own path if needed)
sys.path.append('/xxx/data_synthesis_v2/code')
from llm_generation import LLM_Individual_Generation

# -------------------------- Configuration --------------------------
# Directory for external prompt files
PROMPT_DIR = os.path.join(os.getcwd(), "prompts")

# Base path for all tool modules
TOOL_BASE_PATH = '/xxx/toolenv/v1'

# Mapping for tool classes that require instance initialization
CLASS_BASED_TOOLS = {
    'psyweb_recommender': 'Retriever',
    'story_recommender': 'StoryRetriever',
    'strength_card': 'EncouragingResponder',
    'role_play': 'RoleplayAgent',
    'sticker_respond': 'EmojiRetriever',
    'plan_assistant': 'PlanGenerator'
}

# Multithreading settings
BASE_THREAD_NUM = 20
USERS_PER_THREAD = 5
THREAD_LOCK = threading.Lock()

# -------------------------- Basic Utility Functions --------------------------
def load_json_file(file_path: str) -> Dict:
    """Load and return data from a JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prompt_file(file_name: str) -> str:
    """Load prompt content from a .txt file in the PROMPT_DIR directory"""
    prompt_file_path = os.path.join(PROMPT_DIR, file_name)
    if not os.path.exists(prompt_file_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")
    
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def collect_tool_definitions(base_path: str = TOOL_BASE_PATH) -> Dict:
    """Collect tool definitions from all registered tool directories"""
    tool_definitions = {}
    tool_directories = [
        'joke_recommendation',
        'medical_appointment',
        'music_recommendation',
        'online_shopping',
        'plan_assistant',
        'psyweb_recommender',
        'role_play',
        'schedule_manager',
        'sticker_respond',
        'story_recommender',
        'strength_card',
        'film_recommendation'
    ]

    for tool_dir in tool_directories:
        tool_path = os.path.join(base_path, tool_dir)
        def_path = os.path.join(tool_path, 'definition.json')
        if os.path.exists(def_path):
            tool_definitions[tool_dir] = {
                "definition": load_json_file(def_path),
                "path": tool_path
            }

    return tool_definitions

# -------------------------- LLM Generation Classes --------------------------
class ToolSelectionGeneration(LLM_Individual_Generation):
    """LLM class for generating tool selection and calls"""
    def __init__(self, users_dict, utterance, tool_defs, call_type,
                 suggestion, history, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_dict = users_dict
        self.history = history
        self.tool_defs = tool_defs
        self.utterance = utterance
        self.call_type = call_type
        self.suggestion = suggestion
        self.data_synthesis_root = '/xxx/evaluation/history_based'

    def get_prompt(self, sample_id):
        """Load and format system and user prompts"""
        prompt_dir = os.path.join(self.data_synthesis_root, 'prompt')

        with open(os.path.join(prompt_dir, 'system_tool_selection.txt'), encoding="utf-8") as f:
            system_prompt = ''.join(f.readlines())

        with open(os.path.join(prompt_dir, 'user_tool_selection.txt'), encoding="utf-8") as f:
            user_prompt_template = ''.join(f.readlines())

        user_prompt = user_prompt_template.replace(
            '<persona>', json.dumps(self.users_dict[sample_id]['demographic'], ensure_ascii=False, indent=4))
        user_prompt = user_prompt.replace(
            '<utterance>', json.dumps(self.utterance, ensure_ascii=False, indent=4))
        user_prompt = user_prompt.replace(
            '<tool_defs>', json.dumps(self.tool_defs, ensure_ascii=False, indent=4))
        user_prompt = user_prompt.replace('<history>', '')

        return system_prompt, user_prompt

    def check_generation(self, sample_id, raw_generation):
        """Validate LLM output format and required fields"""
        gen_str = self.generation_postprocess(raw_generation)
        gen_json = json.loads(gen_str)
        required_fields = ['scenario', '1']

        missing_fields = [f for f in required_fields if f not in gen_json]
        if missing_fields:
            return False, f"Missing fields: {', '.join(missing_fields)}"
        return True, None


class FeedbackGenerationGeneration(LLM_Individual_Generation):
    """LLM class for generating natural language feedback from tool results"""
    def __init__(self, persona, user_utterance, tool_results, tool_call, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.persona = persona
        self.user_utterance = user_utterance
        self.tool_results = tool_results
        self.tool_call = tool_call
        self.data_synthesis_root = '/xxx/evaluation/history_based'

    def get_prompt(self, sample_id):
        """Load and format feedback generation prompts"""
        prompt_dir = os.path.join(self.data_synthesis_root, 'prompt_template')

        with open(os.path.join(prompt_dir, 'system_feedback_generation.txt'), encoding="utf-8") as f:
            system_prompt = ''.join(f.readlines())

        with open(os.path.join(prompt_dir, 'user_feedback.txt'), encoding="utf-8") as f:
            user_prompt_template = ''.join(f.readlines())
        user_prompt = user_prompt_template.replace(
            '<persona>', json.dumps(self.persona, ensure_ascii=False, indent=4))
        user_prompt = user_prompt_template.replace(
            '<user_utterance>', json.dumps(self.user_utterance, ensure_ascii=False, indent=4))
        user_prompt = user_prompt.replace(
            '<tool_results>', json.dumps(self.tool_results, ensure_ascii=False, indent=4))
        user_prompt = user_prompt.replace(
            '<tool_call>', json.dumps(self.tool_call, ensure_ascii=False, indent=4))

        return system_prompt, user_prompt

    def check_generation(self, sample_id, raw_generation):
        """Validate feedback output format"""
        generation = self.generation_postprocess(raw_generation)
        generation_dict = json.loads(generation)

        if list(generation_dict.keys()) != ["feedback"]:
            return False, "Invalid key structure"

        for key in generation_dict:
            content = generation_dict[key].strip()
            if not content or content == "...":
                return False, "Empty feedback content"

        return True, None

# -------------------------- Tool Execution Functions --------------------------
def load_tool_module(tool_dir: str, tool_base_path: str = TOOL_BASE_PATH):
    """Dynamically load the api.py module for a given tool directory"""
    api_path = os.path.join(tool_base_path, tool_dir, 'api.py')
    spec = importlib.util.spec_from_file_location(f"{tool_dir}.api", api_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute_tool_call(tool_dir: str, function_name: str, params: Dict):
    """Execute tool function (supports both class methods and regular functions)"""
    module = load_tool_module(tool_dir)

    # Handle class-based tools
    if tool_dir in CLASS_BASED_TOOLS:
        class_name = CLASS_BASED_TOOLS[tool_dir]
        tool_instance = getattr(module, class_name)()
        result = getattr(tool_instance, function_name)(**params)
        return True, result

    # Handle regular function-based tools
    result = getattr(module, function_name)(**params)
    return True, result


def parse_tool_call(call_data: dict) -> tuple:
    """Parse structured tool call data into components"""
    tool_dir = call_data["tool_folder_name"]
    function_name = call_data["function_name"]
    params = call_data["param"]
    return tool_dir, function_name, params, None

# -------------------------- Core Processing Functions --------------------------
def process_single_utterance(users_dict: Dict, utterance, scenario,
                             tool_defs: Dict, call_sequence: int,
                             history, model) -> Dict:
    """Process one user utterance: generate tool call → execute → generate feedback"""
    # Clean user data 
    users_clean = {"0000": users_dict['demographic']}
    users_full = {"0000": users_dict}

    # Save paths
    base_save = '/xxx/evaluation/history_based/data'
    save_selection = os.path.join(base_save, 'debug', 'generate_tool_use', 'selection')
    save_feedback = os.path.join(base_save, 'debug', 'generate_tool_use', 'feedback')

    # Step 1: Generate tool call
    tool_generator = ToolSelectionGeneration(
        users_dict=users_clean, utterance=utterance, tool_defs=tool_defs,
        call_type='pos', suggestion=[], history=history,
        model_name=model, save_dir=save_selection
    )
    success, calls = tool_generator.generate(sample_id='0000')

    if not success:
        return {
            "user_utterance": utterance,
            "tool_call_failed": calls
        }

    # Step 2: Parse and execute tool call (max 3 retries)
    tool_calls = json.loads(calls)
    scenario = tool_calls["scenario"]
    tool_result = {
        "call_sequence": call_sequence,
        "status": "error",
        "result_details": ""
    }
    execute_success = False
    retry_success = False

    for attempt in range(3):
        tool_call = tool_calls["1"]
        tool_dir, func_name, params, _ = parse_tool_call(tool_call)

        if tool_dir not in tool_defs:
            tool_result["result_details"] = f"Tool not found: {tool_dir}"
            continue

        exec_success, exec_result = execute_tool_call(tool_dir, func_name, params)
        tool_result["result_details"] = str(exec_result)

        if exec_success:
            retry_success = True
            if attempt == 0:
                execute_success = True
            break

    # Step 3: Generate natural language feedback
    feedback_generator = FeedbackGenerationGeneration(
        persona=users_clean,
        user_utterance=utterance,
        tool_results=tool_result["result_details"],
        tool_call=tool_call,
        model_name=model,
        save_dir=save_feedback
    )
    fb_success, feedback = feedback_generator.generate(sample_id='0000')

    if not fb_success:
        feedback = f"Feedback generation failed: {feedback}"

    # Build final result
    return {
        "user_utterance": utterance,
        "scenario": scenario,
        "tool_call": {
            "1": {
                "tool_call": tool_call,
                "tool_raw_result": tool_result["result_details"],
                "generated_feedback": feedback,
                "execute_success": execute_success,
                "3turn_execute_success": retry_success,
                "feedback_success": fb_success
            }
        }
    }


def check_valid_toolcall(tool_call, users_dict: Dict, utterance,
                         tool_defs: Dict, call_sequence: int, model):
    """Check if a tool call can be executed successfully"""
    tool_dir, func_name, params, _ = parse_tool_call(tool_call)
    if tool_dir not in tool_defs:
        return False
    success, _ = execute_tool_call(tool_dir, func_name, params)
    return success


def single_toolcall_generation(tool_call, users_dict: Dict, utterance,
                               tool_defs: Dict, call_sequence: int, model):
    """Execute a single pre-defined tool call and generate feedback"""
    base_save = '/xxx/evaluation/profile_based/data'
    save_feedback = os.path.join(base_save, 'debug', 'generate_tool_use', 'feedback')

    tool_dir, func_name, params, _ = parse_tool_call(tool_call)
    exec_success, exec_result = execute_tool_call(tool_dir, func_name, params)

    # Generate feedback
    feedback_generator = FeedbackGenerationGeneration(
        persona=users_dict['demographic'],
        user_utterance=utterance,
        tool_results=str(exec_result),
        tool_call=tool_call,
        model_name=model,
        save_dir=save_feedback
    )
    fb_success, feedback = feedback_generator.generate(sample_id='0000')

    if not fb_success:
        feedback = f"Feedback generation failed: {feedback}"

    return {
        "user_utterance": utterance,
        "scenario": "",
        "tool_call": {
            "1": {
                "tool_call": tool_call,
                "tool_raw_result": str(exec_result),
                "generated_feedback": feedback,
                "execute_success": exec_success
            }
        }
    }

# -------------------------- Multithreading Processing --------------------------
def process_user_group(user_ids, user_data, amount, tool_defs, model, temp_save_path):
    """Process a group of users in a single thread"""
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] Processing user group: {user_ids}")

    group_result = {}
    for user_id in user_ids:
        data = user_data[user_id]
        utterances = data["utterance"]
        scenarios = data["scenarios"]
        persona = data["persona"]

        interactions = []
        # Process utterances 12-15 (adjust range as needed)
        for i in range(12, min(15, len(utterances))):
            result = process_single_utterance(
                users_dict=persona,
                utterance=utterances[i],
                scenario=scenarios[i],
                tool_defs=tool_defs,
                call_sequence=i+1,
                history=[],
                model=model
            )
            interactions.append(result)
            print(f"[{thread_name}] Processed user {user_id} utterance {i}")

        group_result[user_id] = interactions

    # Thread-safe save to temp file
    with THREAD_LOCK:
        existing = load_json_file(temp_save_path) if os.path.exists(temp_save_path) else {}
        existing.update(group_result)
        with open(temp_save_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"[{thread_name}] User group {user_ids} completed")
    return group_result


def generate_tool_main_multithread(amount, user_info_path, final_save_path, model):
    """Main multithreading entry point"""
    # Load user data
    user_data = load_json_file(user_info_path)
    all_user_ids = list(user_data.keys())
    total_users = len(all_user_ids)
    print(f"Loaded {total_users} users")

    # Adjust thread count based on user count
    thread_num = (total_users + USERS_PER_THREAD - 1) // USERS_PER_THREAD
    print(f"Using {thread_num} threads")

    # Load tool definitions
    tool_defs = collect_tool_definitions()

    # Split users into groups
    user_groups = [all_user_ids[i:i+USERS_PER_THREAD]
                   for i in range(0, total_users, USERS_PER_THREAD)]

    # Temp file for parallel results
    temp_path = f"{final_save_path}.temp"
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Run multithreading
    with ThreadPoolExecutor(max_workers=thread_num, thread_name_prefix="UserProc") as executor:
        futures = [
            executor.submit(process_user_group, group, user_data, amount,
                           tool_defs, model, temp_path)
            for group in user_groups
        ]
        for future in as_completed(futures):
            future.result()

    # Merge final output
    final_data = load_json_file(temp_path)
    output = {"user_interactions": final_data}

    with open(final_save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    os.remove(temp_path)
    print(f"All done! Results saved to: {final_save_path}")

# -------------------------- Main Execution --------------------------
if __name__ == "__main__":
    # Configuration
    UTTERANCES_PER_USER = 15
    USER_DATA_PATH = "/xxx/evaluation/profile_based/data/test_set_restructured.json"

    # Models to evaluate
    MODELS = ['gpt-5.1']

    # Run for each model
    for model in MODELS:
        print(f"\n========== Starting evaluation with model: {model} ==========")
        output_path = f"/xxx/evaluation/history_based/data/tool_invoc_result/tool_interactions_without_his_{model}_re.json"

        generate_tool_main_multithread(
            amount=UTTERANCES_PER_USER,
            user_info_path=USER_DATA_PATH,
            final_save_path=output_path,
            model=model
        )

    print("\nAll models completed successfully!")