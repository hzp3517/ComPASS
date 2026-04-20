"""Final version for generating tool calls for evaluation."""

import json
import os
import importlib.util
from typing import Dict
import sys
sys.path.append('/xx/direction_of_llm_generation')
from llm_generation import LLM_Individual_Generation
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# -------------------------- Configuration --------------------------
PROMPT_DIR = os.path.join(os.getcwd(), "prompts")
TOOL_BASE_PATH = '/xx/Empathetic_Interaction/toolenv/v1'

CLASS_BASED_TOOLS = {
    'psyweb_recommender': 'Retriever',
    'story_recommender': 'StoryRetriever',
    'strength_card': 'EncouragingResponder',
    'role_play': 'RoleplayAgent',
    'sticker_respond': 'EmojiRetriever',
    'plan_assistant': 'PlanGenerator'
}

BASE_THREAD_NUM = 20
USERS_PER_THREAD = 5
LOCK = threading.Lock()


# -------------------------- Utility Functions --------------------------
def load_json_file(file_path: str) -> Dict:
    """Load a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_tool_definitions(base_path: str = TOOL_BASE_PATH) -> Dict:
    """Collect API definitions for all available tools."""
    tool_definitions = {}
    tool_dirs = [
        'joke_recommendation', 'medical_appointment', 'music_recommendation',
        'online_shopping', 'plan_assistant', 'psyweb_recommender', 'role_play',
        'schedule_manager', 'sticker_respond', 'story_recommender',
        'strength_card', 'film_recommendation'
    ]

    for tool_dir in tool_dirs:
        def_path = os.path.join(base_path, tool_dir, 'definition.json')
        if os.path.exists(def_path):
            tool_definitions[tool_dir] = {
                "definition": load_json_file(def_path),
                "path": os.path.join(base_path, tool_dir)
            }
    return tool_definitions


# -------------------------- Tool Execution --------------------------
def load_tool_module(tool_dir: str):
    """Load api.py from a tool directory."""
    api_path = os.path.join(TOOL_BASE_PATH, tool_dir, 'api.py')
    if not os.path.exists(api_path):
        return None, f"api.py not found in {tool_dir}"

    try:
        spec = importlib.util.spec_from_file_location(f"{tool_dir}.api", api_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, None
    except Exception as e:
        return None, str(e)


def execute_tool_call(tool_dir: str, function_name: str, params: Dict):
    """Execute a tool call."""
    module, error = load_tool_module(tool_dir)
    if not module:
        return False, error

    try:
        if tool_dir in CLASS_BASED_TOOLS:
            class_obj = getattr(module, CLASS_BASED_TOOLS[tool_dir])()
            func = getattr(class_obj, function_name)
        else:
            func = getattr(module, function_name)

        return True, func(**params)

    except Exception as e:
        return False, str(e)


def parse_tool_call(call: Dict):
    """Parse tool call dictionary."""
    try:
        return call["tool_folder_name"], call["function_name"], call["param"], None
    except Exception as e:
        return None, None, None, str(e)


# -------------------------- LLM Generation --------------------------
class ToolSelectionGeneration(LLM_Individual_Generation):
    """Generate tool selection."""

    def __init__(self, users_dict, utterance, tool_defs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_dict = users_dict
        self.utterance = utterance
        self.tool_defs = tool_defs

    def check_generation(self, sample_id, raw_generation):
        try:
            data = json.loads(self.generation_postprocess(raw_generation))
            return 'scenario' in data and '1' in data, None
        except:
            return False, "Invalid JSON"


class FeedbackGeneration(LLM_Individual_Generation):
    """Generate feedback."""

    def check_generation(self, sample_id, raw_generation):
        try:
            data = json.loads(self.generation_postprocess(raw_generation))
            return list(data.keys()) == ['feedback'], None
        except:
            return False, "Invalid JSON"


# -------------------------- Core Processing --------------------------
def process_single_utterance(user, utterance, tool_defs, model):
    """Process one utterance."""

    generator = ToolSelectionGeneration(
        users_dict={"0000": user},
        utterance=utterance,
        tool_defs=tool_defs,
        model_name=model
    )

    success, result = generator.generate(sample_id='0000')
    if not success:
        return {"error": result}

    calls = json.loads(result)
    tool_call = calls["1"]

    tool_dir, func, params, err = parse_tool_call(tool_call)

    if err or tool_dir not in tool_defs:
        return {"error": err or "Invalid tool"}

    success, tool_result = execute_tool_call(tool_dir, func, params)

    return {
        "utterance": utterance,
        "tool_call": tool_call,
        "tool_result": str(tool_result),
        "success": success
    }


# -------------------------- Multi-threading --------------------------
def process_user_group(user_ids, user_data, amount, tool_defs, model, temp_path):
    thread = threading.current_thread().name
    print(f"[{thread}] Processing users: {user_ids}")

    results = {}

    for uid in user_ids:
        if uid not in user_data:
            continue

        user = user_data[uid]
        utterances = user["utterance"][:amount]

        results[uid] = [
            process_single_utterance(user["persona"], u, tool_defs, model)
            for u in utterances
        ]

    with LOCK:
        if os.path.exists(temp_path):
            existing = load_json_file(temp_path)
        else:
            existing = {}

        existing.update(results)

        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"[{thread}] Finished users: {user_ids}")


def generate_tool_main_multithread(amount, user_path, save_path, model):
    user_data = load_json_file(user_path)
    tool_defs = collect_tool_definitions()

    user_ids = list(user_data.keys())

    groups = [
        user_ids[i:i+USERS_PER_THREAD]
        for i in range(0, len(user_ids), USERS_PER_THREAD)
    ]

    temp_path = save_path + ".temp"
    if os.path.exists(temp_path):
        os.remove(temp_path)

    with ThreadPoolExecutor(max_workers=BASE_THREAD_NUM) as executor:
        futures = [
            executor.submit(process_user_group, g, user_data, amount, tool_defs, model, temp_path)
            for g in groups
        ]

        for f in as_completed(futures):
            f.result()

    final = load_json_file(temp_path)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({"user_interactions": final}, f, indent=2, ensure_ascii=False)

    os.remove(temp_path)

    print("Processing completed successfully.")


# -------------------------- Entry --------------------------
if __name__ == "__main__":
    generate_tool_main_multithread(
        amount=15,
        user_path="/data/.../test.json",
        save_path="/data/.../output.json",
        model="gpt-5.1"
    )
