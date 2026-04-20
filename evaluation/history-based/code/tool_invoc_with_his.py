"""
Tool Call Generation for Evaluation - with history version
"""
import json
import os
import importlib.util
from typing import Dict
import sys
import random
import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Custom module path (update for your environment)
sys.path.append('/xxx/data_synthesis/code')
from llm_generation import LLM_Individual_Generation

# -------------------------- Configuration --------------------------
PROMPT_DIR = os.path.join(os.getcwd(), "prompts")
TOOL_BASE_PATH = '/xxx/toolenv/v1'

# Tool class mapping for instance-based tools
CLASS_BASED_TOOLS = {
    'psyweb_recommender': 'Retriever',
    'story_recommender': 'StoryRetriever',
    'strength_card': 'EncouragingResponder',
    'role_play': 'RoleplayAgent',
    'sticker_respond': 'EmojiRetriever',
    'plan_assistant': 'PlanGenerator'
}

# Multi-threading settings
BASE_THREAD_NUM = 1
USERS_PER_THREAD = 100
THREAD_LOCK = threading.Lock()

# -------------------------- Utility Functions --------------------------
def load_json_file(file_path: str) -> Dict:
    """Load and parse JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prompt_file(file_name: str) -> str:
    """Load prompt text from external file"""
    prompt_file_path = os.path.join(PROMPT_DIR, file_name)
    if not os.path.exists(prompt_file_path):
        raise FileNotFoundError(f"Prompt file missing: {prompt_file_path}")
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def collect_tool_definitions(base_path: str = TOOL_BASE_PATH) -> Dict:
    """Collect tool metadata and definitions from all tool directories"""
    tool_definitions = {}
    tool_dirs = [
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

    for tool_dir in tool_dirs:
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
    """LLM wrapper for generating tool calls from user input"""
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
        prompt_template_dir = os.path.join(self.data_synthesis_root, 'prompt_template')

        with open(os.path.join(prompt_template_dir, 'system_tool_selection.txt'), encoding="utf-8") as f:
            system_prompt = ''.join(f.readlines())

        if self.call_type == 'pos':
            with open(os.path.join(prompt_template_dir, 'user_tool_selection.txt'), encoding="utf-8") as f:
                user_prompt_template = ''.join(f.readlines())

        user_prompt = user_prompt_template.replace(
            '<persona>', json.dumps(self.users_dict[sample_id]['demographic'], indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace(
            '<utterance>', json.dumps(self.utterance, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace(
            '<tool_defs>', json.dumps(self.tool_defs, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace(
            '<history>', json.dumps(self.history, indent=4, ensure_ascii=False))

        return system_prompt, user_prompt

    def check_generation(self, sample_id, raw_generation):
        gen_str = self.generation_postprocess(raw_generation)
        gen_json = json.loads(gen_str)
        required_fields = ['scenario', '1']
        missing_fields = [f for f in required_fields if f not in gen_json]

        if missing_fields:
            return False, f"Missing fields: {', '.join(missing_fields)}"
        return True, None


class FeedbackGenerationGeneration(LLM_Individual_Generation):
    """LLM wrapper for generating natural language feedback from tool results"""
    def __init__(self, persona, user_utterance, tool_results, tool_call, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.persona = persona
        self.user_utterance = user_utterance
        self.tool_results = tool_results
        self.tool_call = tool_call
        self.data_synthesis_root = '/xxx/evaluation/history_based'

    def get_prompt(self, sample_id):
        prompt_template_dir = os.path.join(self.data_synthesis_root, 'prompt')

        with open(os.path.join(prompt_template_dir, 'system_feedback_generation.txt'), encoding="utf-8") as f:
            system_prompt = ''.join(f.readlines())

        with open(os.path.join(prompt_template_dir, 'user_feedback.txt'), encoding="utf-8") as f:
            user_prompt_template = ''.join(f.readlines())
        user_prompt = user_prompt_template.replace(
            '<persona>', json.dumps(self.persona, indent=4, ensure_ascii=False))
        user_prompt = user_prompt_template.replace(
            '<user_utterance>', json.dumps(self.user_utterance, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace(
            '<tool_results>', json.dumps(self.tool_results, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace(
            '<tool_call>', json.dumps(self.tool_call, indent=4, ensure_ascii=False))

        return system_prompt, user_prompt

    def check_generation(self, sample_id, raw_generation):
        generation = self.generation_postprocess(raw_generation)
        generation_dict = json.loads(generation)

        if list(generation_dict.keys()) != ["feedback"]:
            return False, "Invalid key structure"

        for k in generation_dict:
            content = generation_dict[k].strip()
            if not content or content == "...":
                return False, "Empty feedback content"

        return True, None


class PreferGeneration(LLM_Individual_Generation):
    """LLM wrapper for generating user preference feedback"""
    def __init__(self, prefer, utterance, scenario, tool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefer = prefer
        self.scenario = scenario
        self.utterance = utterance
        self.tool_result = tool
        self.data_synthesis_root = '/xxx/data_synthesis'

    def get_prompt(self, sample_id):
        prompt_template_dir = os.path.join(self.data_synthesis_root, 'prompt', 'long_sequence_record')

        with open(os.path.join(prompt_template_dir, 'system_prompt_perfer.txt'), encoding="utf-8") as f:
            system_prompt = ''.join(f.readlines())

        with open(os.path.join(prompt_template_dir, 'user_prompt_perfer.txt'), encoding="utf-8") as f:
            user_prompt_template = ''.join(f.readlines())

        user_prompt = user_prompt_template.replace(
            '<perfer>', json.dumps(self.prefer, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace(
            '<scenario>', json.dumps(self.scenario, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace('<utterance>', self.utterance)
        user_prompt = user_prompt.replace(
            '<tool_result>', json.dumps(self.tool_result, indent=4, ensure_ascii=False))

        return system_prompt, user_prompt

    def check_generation(self, sample_id, raw_generation):
        generation = self.generation_postprocess(raw_generation)
        generation_dict = json.loads(generation)

        if list(generation_dict.keys()) != ['output']:
            return False, "Invalid key structure"

        for k in generation_dict:
            content = generation_dict[k].strip()
            if not content or content == "...":
                return False, "Empty content"

        return True, None

# -------------------------- Tool Execution Module --------------------------
def load_tool_module(tool_dir: str, tool_base_path: str = TOOL_BASE_PATH):
    """Dynamically load tool API module"""
    api_file_path = os.path.join(tool_base_path, tool_dir, 'api.py')
    spec = importlib.util.spec_from_file_location(f"{tool_dir}.api", api_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, None


def execute_tool_call(tool_dir: str, function_name: str, params: Dict):
    """Execute tool function (supports both class and function-based tools)"""
    module, error_msg = load_tool_module(tool_dir)

    # Handle class-based tools
    if tool_dir in CLASS_BASED_TOOLS:
        class_name = CLASS_BASED_TOOLS[tool_dir]
        class_obj = getattr(module, class_name)()
        result = getattr(class_obj, function_name)(**params)
        return True, result

    # Handle regular function tools
    result = getattr(module, function_name)(**params)
    return True, result


def parse_tool_call(call_data: dict) -> tuple:
    """Parse structured tool call into components"""
    tool_dir = call_data["tool_folder_name"]
    function_name = call_data["function_name"]
    params = call_data["param"]
    return tool_dir, function_name, params, None


def user_feedback_generate(prefer, utterance, scenario, tool, model):
    """Generate user satisfaction feedback based on tool performance"""
    save_dir = "/xxx/data_synthesis_v2/data/debug/long_sequence_record/perfer"
    generator = PreferGeneration(
        prefer=prefer, tool=tool, utterance=utterance,
        scenario=scenario, model_name=model, save_dir=save_dir
    )

    success, result = generator.generate(sample_id="0")
    if not success:
        return "User feedback generation failed"

    result = json.loads(result)["output"]
    if result == "Like":
        return "I’m very satisfied with this tool for this scenario."
    return "In this situation, I’m quite annoyed with this tool."

# -------------------------- Core Processing Logic --------------------------
def process_single_utterance(
    users_dict: Dict,
    utterance,
    scenario,
    tool_defs: Dict,
    call_sequence: int,
    history,
    model,
    judge_model='gpt-4.1'
) -> Dict:
    """Process one user utterance: generate tool call → execute → generate feedback"""
    users_dict_clean = {"0000": users_dict['demographic']}
    save_root = '/xxx/evaluation/history_based/data'
    save_dir_selection = os.path.join(save_root, 'debug', 'generate_tool_use', 'selection')
    save_dir_feedback = os.path.join(save_root, 'debug', 'generate_tool_use', 'feedback')

    # Generate tool call
    tool_generator = ToolSelectionGeneration(
        users_dict=users_dict_clean,
        utterance=utterance,
        tool_defs=tool_defs,
        call_type='pos',
        suggestion=[],
        history=history,
        model_name=model,
        save_dir=save_dir_selection
    )
    success, calls = tool_generator.generate(sample_id='0000')

    if not success:
        return {
            "user_utterance": utterance,
            "tool_call_failed": calls
        }

    # Execute tool call (max 3 retries)
    attempt = 0
    is_pass = False
    exec_flag = False
    tool_result = {}

    while not is_pass and attempt < 3:
        attempt += 1
        tool_calls = json.loads(calls)
        scenario = tool_calls["scenario"]
        tool_call = tool_calls["1"]

        tool_dir, func_name, params, parse_error = parse_tool_call(tool_call)
        tool_result = {
            "call_sequence": call_sequence,
            "call_content": tool_call,
            "status": "error",
            "result_details": ""
        }

        if parse_error:
            tool_result["result_details"] = parse_error
        elif tool_dir not in tool_defs:
            tool_result["result_details"] = f"Tool folder not exist: {tool_dir}"
        else:
            exec_success, exec_result = execute_tool_call(tool_dir, func_name, params)
            if exec_success:
                is_pass = True
                if attempt == 1:
                    exec_flag = True
            tool_result["result_details"] = str(exec_result)

    # Generate natural language feedback
    feedback_generator = FeedbackGenerationGeneration(
        persona=users_dict_clean,
        user_utterance=utterance,
        tool_results=tool_result["result_details"],
        tool_call=tool_call,
        model_name=model,
        save_dir=save_dir_feedback
    )
    feedback_success, feedback = feedback_generator.generate(sample_id='0000')
    feedback_pass = True

    if not feedback_success:
        feedback = f"Feedback generation failed: {feedback}"
        feedback_pass = False

    # Build result structure
    final_result = {
        "1": {
            "tool_call": tool_call,
            "tool_raw_result": tool_result["result_details"],
            "generated_feedback": feedback,
            "execute_success": exec_flag,
            "3turn_execute_success": is_pass,
            "feedback_success": feedback_pass
        }
    }

    # Generate user satisfaction feedback
    user_feedback = user_feedback_generate(
        prefer=users_dict,
        utterance=utterance,
        scenario=scenario,
        tool=feedback,
        model=judge_model
    )

    # Update conversation history
    record = {
        "user_utterance": utterance,
        "agent_reply": feedback,
        "user_feedback": user_feedback
    }

    return record, {
        "user_utterance": utterance,
        "scenario": scenario,
        "tool_call": final_result,
        "user_feedback": record
    }


def check_valid_toolcall(tool_call, users_dict: Dict, utterance, tool_defs: Dict, call_sequence: int, model):
    """Validate if a tool call can be executed successfully"""
    tool_dir, func_name, params, parse_error = parse_tool_call(tool_call)
    if parse_error or tool_dir not in tool_defs:
        return False
    exec_success, _ = execute_tool_call(tool_dir, func_name, params)
    return exec_success


def single_toolcall_generation(tool_call, users_dict: Dict, utterance, tool_defs: Dict, call_sequence: int, model):
    """Execute a predefined tool call and generate feedback"""
    save_root = '/xxx/evaluation/profile_based/data'
    save_dir_feedback = os.path.join(save_root, 'debug', 'generate_tool_use', 'feedback')

    tool_dir, func_name, params, parse_error = parse_tool_call(tool_call)
    exec_success, exec_result = execute_tool_call(tool_dir, func_name, params)

    # Generate feedback
    feedback_generator = FeedbackGenerationGeneration(
        user_utterance=utterance,
        tool_results=str(exec_result),
        tool_call=tool_call,
        model_name=model,
        save_dir=save_dir_feedback
    )
    feedback_success, feedback = feedback_generator.generate(sample_id='0000')

    if not feedback_success:
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

# -------------------------- Multi-Threading Processing --------------------------
def process_user_group(user_ids, user_data, amount, tool_defs, model, temp_save_path):
    """Process a group of users in a single thread"""
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] Processing user group: {user_ids}")

    group_result = {}
    for user_id in user_ids:
        utterances = user_data[user_id]["utterance"]
        scenarios = user_data[user_id]["scenarios"]
        persona = user_data[user_id]["persona"]

        history = []
        user_interactions = []

        for i in range(amount):
            print(f"Generating for user {user_id} :: utterance {i + 1}")
            record, result = process_single_utterance(
                users_dict=persona,
                utterance=utterances[i],
                scenario=scenarios[i],
                tool_defs=tool_defs,
                call_sequence=i + 1,
                history=history,
                model=model
            )
            history.append(record)
            user_interactions.append(result)
            print(f"[{thread_name}] Completed user {user_id} utterance {i}")

        group_result[user_id] = user_interactions

    # Thread-safe file write
    with THREAD_LOCK:
        existing_data = load_json_file(temp_save_path) if os.path.exists(temp_save_path) else {}
        existing_data.update(group_result)
        with open(temp_save_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

    print(f"[{thread_name}] User group {user_ids} completed")
    return group_result


def generate_tool_main_multithread(amount, user_info_path: str, final_save_path: str, model: str) -> None:
    """Main multi-threaded pipeline entry point"""
    # Load user data
    user_data = load_json_file(user_info_path)
    all_user_ids = list(user_data.keys())
    total_users = len(all_user_ids)
    print(f"Loaded {total_users} users")

    # Adjust thread count
    THREAD_NUM = (total_users + USERS_PER_THREAD - 1) // USERS_PER_THREAD
    print(f"Adjusted thread count: {THREAD_NUM}")

    # Load tool definitions
    tool_defs = collect_tool_definitions()
    if not tool_defs:
        print("No tool definitions loaded. Exiting.")
        return

    # Split users into groups
    user_groups = [all_user_ids[i:i + USERS_PER_THREAD] for i in range(0, total_users, USERS_PER_THREAD)]
    print(f"Split users into {len(user_groups)} groups")

    # Temporary file for parallel results
    temp_save_path = f"{final_save_path}.temp"
    if os.path.exists(temp_save_path):
        os.remove(temp_save_path)

    # Execute multi-thread processing
    with ThreadPoolExecutor(max_workers=THREAD_NUM, thread_name_prefix="UserProcess") as executor:
        futures = [
            executor.submit(process_user_group, group, user_data, amount, tool_defs, model, temp_save_path)
            for group in user_groups
        ]

        for future in as_completed(futures):
            future.result()

    # Merge final results
    all_results = load_json_file(temp_save_path)
    final_output = {"user_interactions": all_results}

    with open(final_save_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    os.remove(temp_save_path)
    print(f"\nProcessing completed. Results saved to: {final_save_path}")
    print(f"Processed {len(all_results)} users")

# -------------------------- Main Entry --------------------------
if __name__ == "__main__":
    # Configuration
    UTTERANCES_PER_USER = 15
    USER_DATA_PATH = "/xxx/evaluation/profile_based/data/test_set_restructured.json"

    # Model list for evaluation
    MODEL_LIST = ['Deepseek-V3.2']

    # Run evaluation for each model
    for model in MODEL_LIST:
        print(f"\n========== STARTING EVALUATION WITH MODEL: {model} ==========")
        final_save_path = f"/xxx/evaluation/history_based/data/tool_invoc_result/tool_interactions_final_{model}.json"

        generate_tool_main_multithread(
            amount=UTTERANCES_PER_USER,
            user_info_path=USER_DATA_PATH,
            final_save_path=final_save_path,
            model=model
        )

    print("\nAll models completed successfully!")