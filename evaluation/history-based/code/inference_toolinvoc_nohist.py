import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import importlib.util
from typing import List, Dict
import sys
import torch
import re
import copy
import time
import logging
from datetime import datetime
from swift.llm import PtEngine, InferRequest, RequestConfig, get_template
import warnings
warnings.filterwarnings("ignore")

# -------------------------- LOGGING --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler('swift_nohistory.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -------------------------- CONFIG --------------------------
MODEL_PATH = "/xxx/LLM-Research/ComPASS-Qwen"
MODEL_NAME = 'compass'
TOOL_BASE_PATH = '/xxx/Empathetic_Interaction/toolenv/'
DATA_ROOT = '/xxx/Empathetic_Interaction/evaluation/history_based'

FUNCTION_MAP = {
    'psychology_websites_recommender': 'recommend_psychology_websites',
    'retrieve_story': 'search_story',
    'strengthcard': 'generate_encouragement',
    'roleplay_agent': 'chat_with_character',
    'retrieve_stickers': 'search_emoji',
    'generate_plan': 'create_plan',
    'shopping_assistant_api': 'shopping_assistant_api',
    'query_music': 'query_music'
}

CLASS_BASED_TOOLS = {
    'psyweb_recommender': 'Retriever',
    'story_recommender': 'StoryRetriever',
    'strength_card': 'EncouragingResponder',
    'role_play': 'RoleplayAgent',
    'sticker_respond': 'EmojiRetriever',
    'plan_assistant': 'PlanGenerator'
}

GENERATION_CONFIG = {
    "max_tokens": 1024,
    "temperature": 0.3,
    "top_p": 0.9,
    "do_sample": True
}

# -------------------------- BASE GENERATOR --------------------------
class LocalLlamaGenerator:
    def __init__(self, engine):
        self.engine = engine
        self.template = get_template(engine.model.model_meta.template, engine.tokenizer)
        self.engine.default_template = self.template

    def get_prompt(self, sample_id):
        raise NotImplementedError

    def extract_json_from_text(self, text):
        text = text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        try:
            start_index = text.find('{')
            end_index = text.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                return text[start_index:end_index + 1]
        except:
            pass
        return text

    def check_generation(self, sample_id, raw_generation):
        return True, None

    def generate(self, sample_id):
        system_prompt, user_prompt = self.get_prompt(sample_id)
        system_prompt += "\n\nIMPORTANT: Output ONLY JSON. Do not explain. Start with '{' and end with '}'."

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
            return is_valid, cleaned_json_text
        except Exception as e:
            logger.error(f"Generation failed for sample {sample_id}: {str(e)}")
            return False, f"Generation error: {str(e)}"

# -------------------------- TOOL SELECTION --------------------------
class ToolSelectionGeneration(LocalLlamaGenerator):
    def __init__(self, engine, persona, utterance, tool_defs, history):
        super().__init__(engine)
        self.persona = persona
        self.utterance = utterance
        self.tool_defs = tool_defs
        self.history = history

    def get_prompt(self, sample_id):
        prompt_dir = os.path.join(DATA_ROOT, 'prompt')
        with open(os.path.join(prompt_dir, 'system_tool_selection.txt'), 'r', encoding="utf-8") as f:
            sys_p = f.read()
        with open(os.path.join(prompt_dir, 'user_tool_selection.txt'), 'r', encoding="utf-8") as f:
            usr_p = f.read()

        usr_p = usr_p.replace('<persona>', json.dumps(self.persona, indent=4, ensure_ascii=False))
        usr_p = usr_p.replace('<utterance>', json.dumps(self.utterance, indent=4, ensure_ascii=False))
        usr_p = usr_p.replace('<tool_defs>', json.dumps(self.tool_defs, indent=4, ensure_ascii=False))
        usr_p = usr_p.replace('<history>', json.dumps(self.history, indent=4, ensure_ascii=False))
        return sys_p, usr_p

# -------------------------- FEEDBACK GENERATION --------------------------
class FeedbackGeneration(LocalLlamaGenerator):
    def __init__(self, engine, persona, user_utterance, tool_results, tool_call):
        super().__init__(engine)
        self.persona = persona
        self.user_utterance = user_utterance
        self.tool_results = tool_results
        self.tool_call = tool_call

    def get_prompt(self, sample_id):
        prompt_dir = os.path.join(DATA_ROOT, 'prompt')
        with open(os.path.join(prompt_dir, 'system_feedback_generation.txt'), 'r', encoding="utf-8") as f:
            sys_p = f.read()
        with open(os.path.join(prompt_dir, 'user_feedback_ver1.0.txt'), 'r', encoding="utf-8") as f:
            usr_p = f.read()

        usr_p = usr_p.replace('<persona>', json.dumps(self.persona, indent=4, ensure_ascii=False))
        usr_p = usr_p.replace('<user_utterance>', json.dumps(self.user_utterance, indent=4, ensure_ascii=False))
        usr_p = usr_p.replace('<tool_results>', json.dumps(self.tool_results, indent=4, ensure_ascii=False))
        usr_p = usr_p.replace('<tool_call>', json.dumps(self.tool_call, indent=4, ensure_ascii=False))
        return sys_p, usr_p

# -------------------------- PREFERENCE JUDGE --------------------------
class PreferGeneration(LocalLlamaGenerator):
    def __init__(self, engine, persona, utterance, scenario, tool_feedback):
        super().__init__(engine)
        self.persona = persona
        self.utterance = utterance
        self.scenario = scenario
        self.tool_feedback = tool_feedback

    def get_prompt(self, sample_id):
        prompt_dir = os.path.join('/xxx/Empathetic_Interaction/data_synthesis_v2', 'prompt', 'long_sequence_record')
        with open(os.path.join(prompt_dir, 'system_prompt_perfer.txt'), 'r', encoding="utf-8") as f:
            sys_p = f.read()
        with open(os.path.join(prompt_dir, 'user_prompt_perfer.txt'), 'r', encoding="utf-8") as f:
            usr_p = f.read()

        usr_p = usr_p.replace('<perfer>', json.dumps(self.persona, indent=4, ensure_ascii=False))
        usr_p = usr_p.replace('<scenario>', json.dumps(self.scenario, indent=4, ensure_ascii=False))
        usr_p = usr_p.replace('<utterance>', self.utterance)
        usr_p = usr_p.replace('<tool_result>', json.dumps(self.tool_feedback, indent=4, ensure_ascii=False))
        return sys_p, usr_p

# -------------------------- TOOL EXECUTION --------------------------
def execute_tool_call(tool_call_dict):
    try:
        tool_dir = tool_call_dict["tool_folder_name"]
        func_name = tool_call_dict["function_name"]
        params = tool_call_dict["param"]
        actual_func_name = FUNCTION_MAP.get(func_name, func_name)
        api_path = os.path.join(TOOL_BASE_PATH, tool_dir, 'api.py')
        spec = importlib.util.spec_from_file_location(f"{tool_dir}.api", api_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if tool_dir in CLASS_BASED_TOOLS:
            obj = getattr(module, CLASS_BASED_TOOLS[tool_dir])()
            return True, getattr(obj, actual_func_name)(**params)
        return True, getattr(module, actual_func_name)(**params)
    except Exception as e:
        return False, str(e)

# -------------------------- PIPELINE --------------------------
def process_utterance_pipeline(engine, persona, utterance, tool_defs, history, seq):
    selector = ToolSelectionGeneration(engine, persona, utterance, tool_defs, history)
    calls = {}
    for tempt in range(3):
        success, call_json = selector.generate(sample_id='0')
        if not success:
            continue

        try:
            calls = json.loads(call_json)
            break
        except json.JSONDecodeError:
            logger.warning(f"JSON parse failed, retry {tempt+1}/3")

    scenario = calls.get("scenario", "unknown")
    tool_call = calls.get("1")
    exec_ok, exec_res = execute_tool_call(tool_call)
    feedback = "failed to extract json"

    for attempt in range(3):
        fb_gen = FeedbackGeneration(engine, persona, utterance, str(exec_res), tool_call)
        fb_ok, feedback_json = fb_gen.generate(sample_id='0')

        if not fb_ok:
            continue

        try:
            feedback = json.loads(feedback_json).get("feedback", "")
            break
        except json.JSONDecodeError:
            logger.warning(f"Feedback JSON parse failed, retry {attempt+1}/3")

    return {
        "scenario": scenario,
        "tool_call": {
            "1": {
                "tool_call": tool_call,
                "tool_raw_result": str(exec_res),
                "generated_feedback": feedback,
                "execute_success": exec_ok
            }
        }
    }, None

def load_json_file(file_path: str) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def collect_tool_definitions(base_path: str = TOOL_BASE_PATH) -> Dict:
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

# -------------------------- MAIN --------------------------
def main():
    amount = 15
    model = MODEL_NAME
    save_path = f"{DATA_ROOT}/data/tool_invoc_result/tool_interactions_nohis_{model}.json"
    user_info_path = "/xxx/Empathetic_Interaction/evaluation/profile_based/data/test_set_restructured.json"

    logger.info("Loading model (SwiftLLM PtEngine)...")

    try:
        engine = PtEngine(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
            load_in_4bit=True
        )
        logger.info(f"Model loaded successfully on device: {engine.model.device}")
        template = get_template(engine.model.model_meta.template, engine.tokenizer)
        engine.default_template = template

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        return

    user_data = json.load(open(user_info_path, 'r'))
    tool_defs = collect_tool_definitions()
    results = {}

    if os.path.exists(save_path):
        results = json.load(open(save_path, 'r')).get("user_interactions", {})

    for u_idx, (user_id, info) in enumerate(user_data.items()):
        if user_id in results and len(results[user_id]) >= amount:
            continue

        logger.info(f"Processing User {u_idx+1}: {user_id}")
        history = []
        user_interactions = []
        persona = info["persona"]
        persona_clean = persona['demographic']

        for i in range(12, 15):
            res, err = process_utterance_pipeline(engine, persona_clean, info["utterance"][i], tool_defs, history, i+1)
            user_interactions.append(res if res else {"error": err})
            logger.info(f"Utterance {i+1} completed")

        results[user_id] = user_interactions

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({"user_interactions": results}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()