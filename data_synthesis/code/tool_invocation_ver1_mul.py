import json
import os
import importlib.util
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import copy
import sys


from llm_generation import LLM_Individual_Generation
from toolcall_check import ToolCheckGeneration

sys.path.append('compass/data_synthesis/code')


BASE_THREAD_NUM = 20
USERS_PER_THREAD = 5
LOCK = threading.Lock()


TOOL_MODULE_CACHE = {}
MODULE_CACHE_LOCK = threading.Lock()


PROMPT_DIR = os.path.join(os.getcwd(), "prompts")
TOOL_BASE_PATH = 'tool'
CLASS_BASED_TOOLS = {
    'psyweb_recommender': 'Retriever',
    'story_recommender': 'StoryRetriever',
    'strength_card': 'EncouragingResponder',
    'role_play': 'RoleplayAgent',
    'sticker_respond': 'EmojiRetriever',
    'Empathetic_respond': 'ES_respond',
    'plan_assistant': 'PlanGenerator'
}


def load_json_file(file_path: str) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_prompt_file(file_name: str) -> str:
    prompt_file_path = os.path.join(PROMPT_DIR, file_name)
    if not os.path.exists(prompt_file_path):
        raise FileNotFoundError(f"Prompt file missing: {prompt_file_path}")
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def collect_tool_definitions(base_path: str = TOOL_BASE_PATH) -> Dict:
    tool_definitions = {}
    tool_dirs = [
        'joke_recommendation', 'medical_appointment', 'music_recommendation',
        'online_shopping', 'plan_assistant', 'psyweb_recommender',
        'role_play', 'schedule_manager', 'sticker_respond',
        'story_recommender', 'strength_card', 'film_recommendation'
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


class ToolSelectionGeneration(LLM_Individual_Generation):
    def __init__(self, users_dict, utterance, tool_defs, call_type, suggestion, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_dict = users_dict
        self.tool_defs = tool_defs
        self.utterance = utterance
        self.call_type = call_type
        self.suggestion = suggestion
        self.data_synthesis_root = 'compass/data_synthesis'

    def get_prompt(self, sample_id):
        prompt_template_dir = os.path.join(self.data_synthesis_root, 'prompt', 'tool_invocation')
        with open(os.path.join(prompt_template_dir, 'system_tool_selection.txt'), 'r', encoding="utf-8") as f:
            system_prompt = ''.join(f.readlines())
        if self.call_type == 'pos':
            with open(os.path.join(prompt_template_dir, 'user_tool_selection_pos.txt'), 'r', encoding="utf-8") as f:
                user_prompt_template = ''.join(f.readlines())
        elif self.call_type == 'neg':
            with open(os.path.join(prompt_template_dir, 'user_tool_selection_neg.txt'), 'r', encoding="utf-8") as f:
                user_prompt_template = ''.join(f.readlines())

        user_prompt = user_prompt_template.replace('<persona>', json.dumps(self.users_dict[sample_id], indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace('<utterance>', json.dumps(self.utterance, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace('<tool_defs>', json.dumps(self.tool_defs, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace('<suggestion>', json.dumps(self.suggestion, indent=4, ensure_ascii=False))
        return system_prompt, user_prompt

    def check_generation(self, sample_id, raw_generation):
        gen_str = self.generation_postprocess(raw_generation)
        try:
            gen_json = json.loads(gen_str)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}"

        required_fields = ['scenario', '1', '2']
        missing_fields = [f for f in required_fields if f not in gen_json]
        if missing_fields:
            return False, f"Missing fields: {', '.join(missing_fields)}"
        return True, None


class FeedbackGenerationGeneration(LLM_Individual_Generation):
    def __init__(self, user_utterance, tool_results, tool_call, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_utterance = user_utterance
        self.tool_results = tool_results
        self.tool_call = tool_call
        self.data_synthesis_root = 'compass/data_synthesis'

    def get_prompt(self, sample_id):
        prompt_template_dir = os.path.join(self.data_synthesis_root, 'prompt', 'tool_invocation')
        with open(os.path.join(prompt_template_dir, 'system_feedback_generation.txt'), 'r', encoding="utf-8") as f:
            system_prompt = ''.join(f.readlines())
        with open(os.path.join(prompt_template_dir, 'user_feedback_ver3.0.txt'), 'r', encoding="utf-8") as f:
            user_prompt_template = ''.join(f.readlines())

        user_prompt = user_prompt_template.replace('<user_utterance>', json.dumps(self.user_utterance, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace('<tool_results>', json.dumps(self.tool_results, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace('<tool_call>', json.dumps(self.tool_call, indent=4, ensure_ascii=False))
        return system_prompt, user_prompt

    def check_generation(self, sample_id, raw_generation):
        generation = self.generation_postprocess(raw_generation)
        try:
            generation_dict = json.loads(generation)
            if list(generation_dict.keys()) != ["feedback"]:
                return False, "wrong key"
            for k in list(generation_dict.keys()):
                if generation_dict[k].strip() == "" or generation_dict[k] == "...":
                    return False, "empty requirement"
        except:
            return False, "JSON format error"
        return True, None


class PerferCheckGeneration(LLM_Individual_Generation):
    def __init__(self, users_perfer, scenario, tool_call, type1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_perfer = users_perfer
        self.scenario = scenario
        self.tool_call = tool_call
        self.type1 = type1
        self.data_synthesis_root = 'compass/data_synthesis'

    def get_prompt(self, sample_id):
        prompt_template_dir = os.path.join(self.data_synthesis_root, 'prompt', 'tool_invocation')
        with open(os.path.join(prompt_template_dir, 'system_prompt_check_perfer.txt'), 'r', encoding="utf-8") as f:
            system_prompt = ''.join(f.readlines())
        if self.type1 == "pos":
            with open(os.path.join(prompt_template_dir, 'user_prompt_check_perfer_pos.txt'), 'r', encoding="utf-8") as f:
                user_prompt_template = ''.join(f.readlines())
        elif self.type1 == "neg":
            with open(os.path.join(prompt_template_dir, 'user_prompt_check_perfer_neg.txt'), 'r', encoding="utf-8") as f:
                user_prompt_template = ''.join(f.readlines())

        user_prompt = user_prompt_template.replace('<users_perfer>', json.dumps(self.users_perfer, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace('<scenario>', json.dumps(self.scenario, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace('<tool_call>', json.dumps(self.tool_call, indent=4, ensure_ascii=False))
        return system_prompt, user_prompt

    def check_generation(self, sample_id, raw_generation):
        generation = self.generation_postprocess(raw_generation)
        try:
            generation_dict = json.loads(generation)
            if list(generation_dict.keys()) != ['judge']:
                return False, "wrong key"
            for k in list(generation_dict.keys()):
                if generation_dict[k].strip() == "" or generation_dict[k] == "...":
                    return False, "empty requirement"
        except:
            return False, "JSON format error"
        return True, None


def load_tool_module(tool_dir: str, tool_base_path: str = TOOL_BASE_PATH):
    with MODULE_CACHE_LOCK:
        if tool_dir in TOOL_MODULE_CACHE:
            return TOOL_MODULE_CACHE[tool_dir], None

        api_file_path = os.path.join(tool_base_path, tool_dir, 'api.py')
        if not os.path.exists(api_file_path):
            return None, f"api.py not found in tool folder: {tool_dir}"

        try:
            spec = importlib.util.spec_from_file_location(f"{tool_dir}.api", api_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            TOOL_MODULE_CACHE[tool_dir] = module
            return module, None
        except Exception as e:
            return None, f"Load api.py failed for {tool_dir}: {str(e)}"

def execute_tool_call(tool_dir: str, function_name: str, params: Dict):
    module, error_msg = load_tool_module(tool_dir)
    if not module:
        return False, error_msg

    if tool_dir in CLASS_BASED_TOOLS:
        class_name = CLASS_BASED_TOOLS[tool_dir]
        if not hasattr(module, class_name):
            return False, f"Class {class_name} not found in {tool_dir}"
        try:
            class_obj = getattr(module, class_name)()
            if not hasattr(class_obj, function_name):
                return False, f"Method {function_name} not found"
            return True, getattr(class_obj, function_name)(**params)
        except Exception as e:
            return False, f"Class method call failed: {str(e)}"
    else:
        if not hasattr(module, function_name):
            return False, f"Function {function_name} not found"
        try:
            return True, getattr(module, function_name)(**params)
        except Exception as e:
            return False, f"Normal function call failed: {str(e)}"

def parse_tool_call(call_str: str) -> tuple:
    try:
        tool_dir = call_str["tool_folder_name"]
        function_name = call_str["function_name"]
        params = call_str["param"]
        return tool_dir, function_name, params, None
    except Exception as e:
        return None, None, None, f"Parse tool call failed: {str(e)}"


def process_single_utterance(
    user_id: str,
    users_dict: Dict,
    utterance,
    tool_defs: Dict,
    call_sequence: int,
    model
) -> Dict:
    users_dict_wrapper = {user_id: users_dict, "0000": users_dict}
    data_synthesis_save_root = 'compass/data_synthesis/data'
    save_dir_selection = os.path.join(data_synthesis_save_root, 'debug', 'generate_tool_use','selection')
    save_dir_feedback = os.path.join(data_synthesis_save_root, 'debug', 'generate_tool_use','feedback')
    save_dir_check = os.path.join(data_synthesis_save_root, 'debug', 'generate_tool_use','check')
    save_dir_perfer = os.path.join(data_synthesis_save_root, 'debug', 'generate_tool_use','perfer')

    final_result = {'pos': {}, 'neg': {}}
    scenario = ''
    for i in range(2):
        call_type = ['pos', 'neg'][i]
        Pass = False
        total_suggestion = ""
        time = 0

        while not Pass and time < 1:
            tool_generator = ToolSelectionGeneration(users_dict=users_dict_wrapper, utterance=utterance, tool_defs=tool_defs, call_type=call_type, suggestion=total_suggestion, model_name=model, save_dir=save_dir_selection)
            success, calls = tool_generator.generate(sample_id=user_id)

            if not success:
                return {"user_utterance": utterance, "tool_call_all": f"{calls}"}

            tool_calls = json.loads(calls)
            is_pass = True
            suggestions = []
            scenario = tool_calls.get("scenario", "")

            for key in ["1","2"]:
                if key not in tool_calls: continue
                tool_call = tool_calls[key]
                tool_result = {
                    "call_sequence": call_sequence,
                    "call_content": tool_call,
                    "status": "error",
                    "result_details": ""
                }
                tool = copy.deepcopy(tool_call)
                if 'call_type' in tool:
                    tool.pop('call_type')

                check_perfer_generator = PerferCheckGeneration(users_perfer=users_dict_wrapper[user_id]['tool_preferences'], scenario=scenario, tool_call=tool, type1=call_type, save_dir=save_dir_perfer)
                success, result = check_perfer_generator.generate(sample_id=user_id)
                if success:
                    result = json.loads(result)
                else:
                    result = {}

                if tool_call:
                    tool_dir, func_name, params, parse_error = parse_tool_call(tool_call)
                    if parse_error:
                        tool_result["result_details"] = parse_error
                    elif tool_dir not in tool_defs:
                        tool_result["result_details"] = f"Tool folder not exist: {tool_dir}"
                    else:
                        exec_success, exec_result = execute_tool_call(tool_dir, func_name, params)
                        if exec_success:
                            tool_result["status"] = "success"
                        tool_result["result_details"] = str(exec_result)
                else:
                    tool_result["result_details"] = "No valid tool call generated"

                feedback_generator = FeedbackGenerationGeneration(utterance, tool_result["result_details"], tool, model_name=model, save_dir=save_dir_feedback)
                success, feedback = feedback_generator.generate(sample_id=user_id)
                if success:
                    feedback = json.loads(feedback).get("feedback", "")
                else:
                    feedback = ""

                final_result[call_type][key] = {
                    "type": call_type,
                    "tool_call": tool_call,
                    "tool_result": tool_result["result_details"],
                    "generated_feedback": feedback,
                    "perfer_check": result.get("judge", "Yes") if isinstance(result, dict) else "Yes"
                }

                check_generator = ToolCheckGeneration(users_dict=users_dict_wrapper, user_utterance=utterance, tool_generation=final_result[call_type][key], user_scenario="", type=call_type, model_name="gpt-4o", save_dir=save_dir_check)
                success, check = check_generator.generate(sample_id=user_id)
                if success:
                    check = json.loads(check)
                    score = check.get('score', 0)
                    is_pass = (int(score) > 5) and is_pass
                    final_result[call_type][key]["Pass_check"] = f"{(int(score)>5)}:{score}"
                    final_result[call_type][key]["reason"] = check.get('reasons', '')
                    suggestions.append(check.get("reasons", ""))

            Pass = is_pass
            total_suggestion = suggestions
            time += 1

    return {
        "user_utterance": utterance,
        "scenario": scenario,
        "tool_call_pos": final_result['pos'],
        "tool_call_neg": final_result['neg']
    }


def process_user_chunk(user_keys: List[str], amount: int, persona_dict: Dict, discourses: Dict, tool_defs: Dict, model: str, shared_result_dict: Dict):
    local_dict = {}
    for key in user_keys:
        utterances = discourses.get(key, {})
        persona = persona_dict.get(key, {})

        if len(utterances) < amount:
            print(f"Warning: User {key} has fewer than {amount} utterances, skipped.")
            continue

        utterances_pos = [utterances[f"utterance{i+1}"] for i in range(0, amount // 2)]
        utterances_neg = [utterances[f"utterance{i+1}"] for i in range(amount // 2, amount)]
        all_utterances = utterances_pos + utterances_neg

        user_interactions = []
        for i in range(amount):
            try:
                result = process_single_utterance(
                    user_id=key,
                    users_dict=persona,
                    utterance=all_utterances[i],
                    tool_defs=tool_defs,
                    call_sequence=i+1,
                    model=model
                )
                user_interactions.append(result)
            except Exception as e:
                print(f"Error: Failed to process utterance {i+1} for user {key} -> {str(e)}")

        local_dict[key] = user_interactions
        print(f"Completed: [Thread {threading.current_thread().name}] Processed {amount} utterances for user {key}!")

    with LOCK:
        shared_result_dict.update(local_dict)


def generate_tool_main(
    amount: int,
    user_profile_path: str,
    discourses_path: str,
    save_path: str,
    model: str = "qwen-plus"
) -> None:
    try:
        persona_dict = load_json_file(user_profile_path)
        discourses = load_json_file(discourses_path)
    except Exception as e:
        print(f"Failed to load data files: {str(e)}")
        return

    tool_defs = collect_tool_definitions()
    if not tool_defs:
        print("No tool definitions loaded, exiting.")
        return

    user_keys = list(persona_dict.keys())
    chunks = [user_keys[i:i + USERS_PER_THREAD] for i in range(0, len(user_keys), USERS_PER_THREAD)]
    shared_result_dict = {}

    print(f"Starting multi-threaded task...")
    print(f"Total users: {len(user_keys)} | Total chunks: {len(chunks)} | Concurrent threads: {BASE_THREAD_NUM}")
    print("=" * 50)

    with ThreadPoolExecutor(max_workers=BASE_THREAD_NUM) as executor:
        futures = []
        for chunk in chunks:
            futures.append(
                executor.submit(
                    process_user_chunk,
                    chunk, amount, persona_dict, discourses, tool_defs, model, shared_result_dict
                )
            )

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in worker thread: {e}")

    print("=" * 50)
    try:
        ordered_result_dict = {key: shared_result_dict[key] for key in user_keys if key in shared_result_dict}
        output_json = {"user_interactions": ordered_result_dict}
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)
        print(f"All tasks completed! Results saved to: \n{save_path}")
    except Exception as e:
        print(f"Failed to save results: {str(e)}")


if __name__ == "__main__":
    model = "gpt-5.1"
    amount = 10
    user_profile_path = "compass/data_synthesis/data/persona/persona.json"
    discourses_path = "compass/data_synthesis/data/utterance/utterance.json"
    save_path = f"compass/data_synthesis/data/tool_use/tool_interactions_{model}3.json"

    generate_tool_main(
        amount=amount,
        user_profile_path=user_profile_path,
        discourses_path=discourses_path,
        save_path=save_path,
        model=model
    )