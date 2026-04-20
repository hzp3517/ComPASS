"""
Single-threaded version of generating tool calls using Local Qwen3-8B
Updated with Retry Logic (3 attempts) and Dual Logging (First vs Best)
Adapted to SwiftLLM PtEngine (replace Hugging Face pipeline)
"""
MODULE_CACHE = {}
CLASS_INSTANCE_CACHE = {}
import json
import os
import importlib.util
from typing import List, Dict
import sys
import torch
import re
import copy
import logging
from datetime import datetime
from swift.llm import PtEngine, InferRequest, RequestConfig, get_template
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"/xxx/Empathetic_Interaction/evaluation/profile_based/logs/qwen_toolgen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

MODEL_PATH = "/xxx/model/ComPASS-Qwen"
MODEL_NAME = 'ComPASS'



TOOL_BASE_PATH = '/xxx/Empathetic_Interaction/toolenv/v1'
sys.path.append('/xxx/Empathetic_Interaction/data_synthesis_v2/code')

PROMPT_DIR = os.path.join(os.getcwd(), "prompts")
CLASS_BASED_TOOLS = {
    'psyweb_recommender': 'Retriever',
    'story_recommender': 'StoryRetriever',
    'strength_card': 'EncouragingResponder',
    'role_play': 'RoleplayAgent',
    'sticker_respond': 'EmojiRetriever',
    'plan_assistant': 'PlanGenerator'
}

# SwiftLLM generation config
GENERATION_CONFIG = {
    "max_tokens": 1024,
    "temperature": 0.3, 
    "top_p": 0.9,
    "do_sample": True
}

class LocalSwiftGenerator:
    def __init__(self, engine):
        self.engine = engine  
        self.template = get_template(engine.model.model_meta.template, engine.tokenizer)
        print(f"\nself template\n:\n\n")
        print(engine.default_template)
        engine.default_template = self.template

    def get_prompt(self, sample_id):
        raise NotImplementedError

    def extract_json_from_text(self, text):
        text = text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match: return match.group(1)
        try:
            start_index = text.find('{')
            end_index = text.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                return text[start_index : end_index + 1]
        except: pass
        return text

    def check_generation(self, sample_id, raw_generation):
        raise NotImplementedError

    def generate(self, sample_id, **kwargs):
        system_prompt, user_prompt = self.get_prompt(sample_id)
        system_prompt += "\n\nIMPORTANT: Output ONLY the JSON object. Do not explain your reasoning. Start with '{' and end with '}'."
        user_prompt += "/no_think"
        print(user_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        

        req_config = RequestConfig(
            max_tokens=GENERATION_CONFIG["max_tokens"],
            temperature=GENERATION_CONFIG["temperature"],
            top_p=GENERATION_CONFIG["top_p"],
        )
        
        try:
            req = InferRequest(messages=messages)
            resp = self.engine.infer([req], req_config)
            raw_text = resp[0].choices[0].message.content.strip()
            
            cleaned_json_text = self.extract_json_from_text(raw_text)
            is_valid, error = self.check_generation(sample_id, raw_text)
            if not is_valid:
                logger.warning(f"Validation failed - {error} | Sample: {sample_id}")
                return False, cleaned_json_text
            return True, cleaned_json_text
        except Exception as e:
            logger.error(f"Generation failed for sample {sample_id}: {str(e)}")
            return False, f"Generation error: {str(e)}"

class ToolSelectionGeneration(LocalSwiftGenerator):
    def __init__(self, users_dict, utterance, tool_defs, call_type, suggestion, engine, *args, **kwargs):
        super().__init__(engine=engine, *args, **kwargs)  
        self.users_dict = users_dict
        self.tool_defs = tool_defs
        self.utterance = utterance
        self.call_type = call_type
        self.suggestion = suggestion
        self.data_synthesis_root = '/xxx/Empathetic_Interaction/data_synthesis_v2'

    def get_prompt(self, sample_id):
        prompt_template_dir = os.path.join(self.data_synthesis_root, 'prompt', 'tool_invocation')
        with open(os.path.join(prompt_template_dir, 'system_tool_selection.txt'), 'r', encoding="utf-8") as f:
            system_prompt = ''.join(f.readlines())
        template_name = 'user_tool_selection_pos.txt' if self.call_type == 'pos' else 'user_tool_selection_neg.txt'
        with open(os.path.join(prompt_template_dir, template_name), 'r', encoding="utf-8") as f:
            user_prompt_template = ''.join(f.readlines())
        user_prompt = user_prompt_template.replace('<persona>', json.dumps(self.users_dict[sample_id], indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace('<utterance>', json.dumps(self.utterance, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace('<tool_defs>', json.dumps(self.tool_defs, indent=4, ensure_ascii=False))
        user_prompt = user_prompt.replace('<suggestion>', json.dumps(self.suggestion, indent=4, ensure_ascii=False))
        return system_prompt, user_prompt

    def check_generation(self, sample_id, raw_generation):
        json_str = self.extract_json_from_text(raw_generation)
        try:
            gen_json = json.loads(json_str)
            required_fields = ['scenario', '1']
            missing_fields = [f for f in required_fields if f not in gen_json]
            if missing_fields: return False, f"Missing fields: {', '.join(missing_fields)}"
            return True, None
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)} | Content start: {json_str[:50]}..."

class FeedbackGenerationGeneration(LocalSwiftGenerator):
    def __init__(self, user_utterance, tool_results, tool_call, engine, *args, **kwargs):
        super().__init__(engine=engine, *args, **kwargs)  
        self.user_utterance = user_utterance
        self.tool_results = tool_results
        self.tool_call = tool_call
        self.data_synthesis_root = '/xxx/Empathetic_Interaction/data_synthesis_v2'

    def get_prompt(self, sample_id):
        prompt_template_dir = os.path.join(self.data_synthesis_root, 'prompt', 'tool_invocation')
        with open(os.path.join(prompt_template_dir, 'system_feedback_generation.txt'), 'r', encoding="utf-8") as f:
            system_prompt = ''.join(f.readlines())
        with open(os.path.join(prompt_template_dir, 'user_feedback_ver1.0.txt'), 'r', encoding="utf-8") as f:
            user_prompt_template = ''.join(f.readlines())
        user_prompt = user_prompt_template.replace('<user_utterance>', json.dumps(self.user_utterance, indent=4, ensure_ascii=False))
        user_prompt = user_prompt_template.replace('<tool_results>', json.dumps(self.tool_results, indent=4, ensure_ascii=False))
        user_prompt = user_prompt_template.replace('<tool_call>', json.dumps(self.tool_call, indent=4, ensure_ascii=False))
        return system_prompt, user_prompt
    
    def check_generation(self, sample_id, raw_generation):
        json_str = self.extract_json_from_text(raw_generation)
        try:
            generation_dict = json.loads(json_str)
            if list(generation_dict.keys()) != ["feedback"]: return False, "wrong key (expected 'feedback')"
            for k in list(generation_dict.keys()):
                if generation_dict[k].strip() == "" or generation_dict[k] == "...": return False, "empty content"
        except: return False, "JSON format error"
        return True, None


def load_json_file(file_path: str) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def collect_tool_definitions(base_path: str = TOOL_BASE_PATH) -> Dict:
    tool_definitions = {}
    tool_dirs = [
        'joke_recommendation', 'medical_appointment', 'music_recommendation',
        'online_shopping', 'plan_assistant', 'psyweb_recommender', 'role_play',
        'schedule_manager', 'sticker_respond', 'story_recommender', 'strength_card',
        'film_recommendation'
    ]
    for tool_dir in tool_dirs:
        tool_path = os.path.join(base_path, tool_dir)
        def_path = os.path.join(tool_path, 'definition.json')
        if os.path.exists(def_path):
            tool_definitions[tool_dir] = {"definition": load_json_file(def_path), "path": tool_path}
    return tool_definitions

def load_tool_module(tool_dir: str, tool_base_path: str = TOOL_BASE_PATH):
    if tool_dir in MODULE_CACHE:
        return MODULE_CACHE[tool_dir], None
    
    api_file_path = os.path.join(tool_base_path, tool_dir, 'api.py')
    if not os.path.exists(api_file_path): return None, f"api.py not found: {tool_dir}"
    try:
        spec = importlib.util.spec_from_file_location(f"{tool_dir}.api", api_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        MODULE_CACHE[tool_dir] = module
        return module, None
    except Exception as e: return None, str(e)

def execute_tool_call(tool_dir: str, function_name: str, params: Dict):
    module, error = load_tool_module(tool_dir)
    if not module: return False, error
    if tool_dir in CLASS_BASED_TOOLS:
        class_name = CLASS_BASED_TOOLS[tool_dir]
        if not hasattr(module, class_name): return False, f"Class {class_name} missing"
        try:
            cache_key = f"{tool_dir}_{class_name}"
            if cache_key not in CLASS_INSTANCE_CACHE:
                CLASS_INSTANCE_CACHE[cache_key] = getattr(module, class_name)()
            obj = CLASS_INSTANCE_CACHE[cache_key]

            if not hasattr(obj, function_name): return False, f"Method {function_name} missing"
            return True, getattr(obj, function_name)(**params)
        except Exception as e: return False, str(e)
    else:
        if not hasattr(module, function_name): return False, f"Func {function_name} missing"
        try: return True, getattr(module, function_name)(**params)
        except Exception as e: return False, str(e)

def parse_tool_call(call_str: str) -> tuple:
    try:
        return call_str["tool_folder_name"], call_str["function_name"], call_str["param"], None
    except Exception as e:
        return None, None, None, str(e)

def _execute_single_pass(users_dict: Dict, utterance, tool_defs: Dict, call_sequence: int, engine) -> Dict:
    users_dict_wrapped = {"0000": users_dict}
    final_result = {}
    scenario = ''
    total_suggestion = []

    tool_generator = ToolSelectionGeneration(
        users_dict=users_dict_wrapped, utterance=utterance, tool_defs=tool_defs,
        call_type='pos', suggestion=total_suggestion, engine=engine
    )
    success, calls = tool_generator.generate(sample_id='0000')
    if not success:
        return {"user_utterance": utterance, "tool_call_all": f"{calls}", "is_execution_success": False}
    
    try:
        tool_calls = json.loads(calls)
        scenario = tool_calls.get("scenario", "")
        
        key = "1"
        if key in tool_calls:
            tool_call = tool_calls[key]
            tool_result = {"call_sequence": call_sequence, "call_content": tool_call, "status": "error", "result_details": ""}
            tool_dir, func_name, params, parse_error = parse_tool_call(tool_call)
            
            exec_success = False
            if parse_error:
                tool_result["result_details"] = parse_error
            elif tool_dir not in tool_defs:
                tool_result["result_details"] = f"Tool folder not exist: {tool_dir}"
            else:
                exec_success, exec_result = execute_tool_call(tool_dir, func_name, params)
                tool_result["status"] = "success" if exec_success else "error"
                tool_result["result_details"] = str(exec_result)

            feedback_generator = FeedbackGenerationGeneration(
                user_utterance=utterance, tool_results=tool_result["result_details"],
                tool_call=tool_call, engine=engine
            )
            feedback_success, feedback = feedback_generator.generate(sample_id='0000')
            if not feedback_success:
                feedback = f"Feedback generation failed: {feedback}"

            final_result[key] = {
                "tool_call": tool_call,
                "tool_raw_result": tool_result["result_details"],
                "generated_feedback": feedback,
                "execute_success": exec_success
            }
            
            return {
                "user_utterance": utterance,
                "scenario": scenario,
                "tool_call": final_result,
                "is_execution_success": exec_success
            }
        else:
            return {"user_utterance": utterance, "tool_call_all": calls, "error": "No tool call '1' found", "is_execution_success": False}
            
    except Exception as e:
        return {"user_utterance": utterance, "error": f"Process logic failed: {str(e)}", "tool_call_all": calls, "is_execution_success": False}

def process_utterance_with_retry(users_dict: Dict, utterance, tool_defs: Dict, call_sequence: int, engine) -> tuple:
    """
    Processor with retry logic (replaced pipe with engine)
    Returns: (first_result, best_result)
    """
    first_result = None
    best_result = None
    execution_success = False
    three_execution_success = False
    
    max_retries = 3
    
    for attempt in range(max_retries):
        current_result = _execute_single_pass(users_dict, utterance, tool_defs, call_sequence, engine)
        logger.info(f"current result:{current_result}")
        if attempt == 0:
            execution_success = current_result["is_execution_success"]
            first_result = copy.deepcopy(current_result)
        
        is_success = current_result.get("is_execution_success", False)
        
        if "is_execution_success" in current_result:
            del current_result["is_execution_success"]
        
        if is_success:
            best_result = current_result
            if attempt > 0:
                logger.info(f"  -> Retry successful at attempt {attempt+1}")
            break
        else:
            logger.warning(f"  -> Attempt {attempt+1} failed/invalid. Retrying...")
            best_result = current_result
    three_execution_success = is_success
    
    if first_result and "is_execution_success" in first_result:
        del first_result["is_execution_success"]

    best_result["execution_success"] = execution_success
    best_result["three_execution_success"] = three_execution_success
    return first_result, best_result

def generate_tool_main_single_thread(amount, user_info_path: str, final_save_path_base: str, engine) -> None:
    """
    Main generation function
    """
    try:
        user_data = load_json_file(user_info_path)
        all_user_ids = list(user_data.keys())
        logger.info(f"Successfully loaded {len(all_user_ids)} user records")
    except Exception as e:
        logger.error(f"Failed to load user data: {str(e)}")
        return
    
    tool_defs = collect_tool_definitions()
    
    save_path_first = final_save_path_base.replace(".json", "_first_attempt.json")
    save_path_best = final_save_path_base
    
    all_results_first = {}
    all_results_best = {}
    
    if os.path.exists(save_path_best):
        try:
            with open(save_path_best, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                if "user_interactions" in existing:
                    all_results_best = existing["user_interactions"]
            if os.path.exists(save_path_first):
                with open(save_path_first, 'r', encoding='utf-8') as f:
                    existing_first = json.load(f)
                    if "user_interactions" in existing_first:
                        all_results_first = existing_first["user_interactions"]
            
            logger.info(f"Loaded history: {len(all_results_best)} users")
        except Exception as e:
            logger.warning(f"Failed to load history data: {str(e)}")

    import time
    start_time = time.time()
    for idx, user_id in enumerate(all_user_ids):
        logger.info(f"Processing user [{idx+1}/{len(all_user_ids)}] ID: {user_id}")
        utterances = user_data[user_id]["utterance"]
        persona = user_data[user_id]["persona"]
        
        user_interactions_first = []
        user_interactions_best = []
        
        for i in range(min(amount, len(utterances))):
            try:
                res_first, res_best = process_utterance_with_retry(
                    users_dict=persona, utterance=utterances[i], tool_defs=tool_defs,
                    call_sequence=i+1, engine=engine
                )
                
                user_interactions_first.append(res_first)
                user_interactions_best.append(res_best)
                
                logger.info(f"  - Utterance {idx+1}::{i+1} completed")
            except Exception as e:
                logger.error(f"  - Critical Error: {e}", exc_info=True)
                err_dict = {"error": str(e)}
                user_interactions_first.append(err_dict)
                user_interactions_best.append(err_dict)
        
        all_results_first[user_id] = user_interactions_first
        all_results_best[user_id] = user_interactions_best

        if (idx + 1) % 1 == 0:
            try:
                with open(save_path_first, 'w', encoding='utf-8') as f:
                    json.dump({"user_interactions": all_results_first}, f, ensure_ascii=False, indent=2)
                
                with open(save_path_best, 'w', encoding='utf-8') as f:
                    json.dump({"user_interactions": all_results_best}, f, ensure_ascii=False, indent=2)
                    
                logger.info(f"  [Saved] Progress: {idx+1}/{len(all_user_ids)}")
            except Exception as e:
                logger.error(f"Failed to save file: {e}", exc_info=True)

    logger.info(f"Completed! Time elapsed: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    amount = 15
    user_info_path = "/xxx/Empathetic_Interaction/evaluation/profile_based/data/test_set_restructured.json"
    model_name = MODEL_NAME
    final_save_path = f"/xxx/Empathetic_Interaction/evaluation/profile_based/data/model_comparison/tool_interactions_final_{model_name}.json"
    
    logger.info("Loading Qwen3-8B fine-tuned model (SwiftLLM)...")
    
    try:
        engine = PtEngine(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
            load_in_4bit=False
        )
        print(engine.model.device)
        template = get_template(engine.model.model_meta.template, engine.tokenizer)
        engine.default_template = template
        
        logger.info("Model loaded successfully (SwiftLLM PtEngine)!")
        
        logger.info(f"\n========== STARTING WITH MODEL: {model_name} (Retry Enabled) ==========")
        generate_tool_main_single_thread(
            amount=amount,
            user_info_path=user_info_path,
            final_save_path_base=final_save_path,
            engine=engine
        )
        
    except Exception as e:
        logger.error(f"Main program failed: {str(e)}", exc_info=True)
        raise