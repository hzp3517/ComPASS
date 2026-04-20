import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import importlib.util
from typing import Dict
import re
import sys
import torch

from swift.llm import PtEngine, InferRequest, RequestConfig, get_template

sys.path.append('/xxx/Empathetic_Interaction/data_synthesis_v2/code')
from llm_generation import LLM_Individual_Generation

# -------------------------- CONFIG --------------------------

LLAMA_MODEL_PATH = "/xxx/LLM-Research/Qwen3-8B-SFT-Merged"
assert os.path.exists(LLAMA_MODEL_PATH), f"Model path not found: {LLAMA_MODEL_PATH}"
MODEL_NAME = 'compass'
TOOL_BASE_PATH = '/xxx/Empathetic_Interaction/toolenv/v1'
DATA_ROOT = '/xxx/Empathetic_Interaction/evaluation/history_based'

GEN_CONFIG = RequestConfig(
    max_tokens=2048,
    temperature=0.3,
    top_p=0.9,
)

CLASS_BASED_TOOLS = {
    'psyweb_recommender': 'Retriever',
    'story_recommender': 'StoryRetriever',
    'strength_card': 'EncouragingResponder',
    'role_play': 'RoleplayAgent',
    'sticker_respond': 'EmojiRetriever',
    'plan_assistant': 'PlanGenerator'
}

# -------------------------- BASE GENERATOR --------------------------

class LocalLlamaGenerator:

    def __init__(self, engine):
        self.engine = engine

    def get_prompt(self, sample_id):
        raise NotImplementedError

    def extract_json_from_text(self, text):
        text = text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)

        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return text[start:end + 1]
        except:
            pass
        return text

    def generate(self, sample_id):
        system_prompt, user_prompt = self.get_prompt(sample_id)
        system_prompt += "\n\nIMPORTANT: Output ONLY JSON."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        req = InferRequest(messages=messages)
        resp = self.engine.infer([req], GEN_CONFIG)[0]
        raw_text = resp.choices[0].message.content
        cleaned_json = self.extract_json_from_text(raw_text)

        return True, cleaned_json

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
        self.user_utterance = user_utterance
        self.tool_results = tool_results
        self.tool_call = tool_call
        self.persona = persona

    def get_prompt(self, sample_id):
        prompt_dir = os.path.join(DATA_ROOT, 'prompt', 'tool_invocation')
        if not os.path.exists(prompt_dir):
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

# -------------------------- GPT JUDGE --------------------------

class PreferGeneration(LLM_Individual_Generation):

    def __init__(self, perfer, utterance, scenario, tool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.perfer = perfer
        self.scenario = scenario
        self.utterance = utterance
        self.tool_result = tool
        self.data_synthesis_root = '/xxx/Empathetic_Interaction/data_synthesis_v2'

    def get_prompt(self, sample_id):
        prompt_dir = os.path.join(
            self.data_synthesis_root, 'prompt', 'long_sequence_record'
        )

        system_prompt = open(
            os.path.join(prompt_dir, 'system_prompt_perfer.txt'), encoding="utf-8"
        ).read()

        user_template = open(
            os.path.join(prompt_dir, 'user_prompt_perfer.txt'), encoding="utf-8"
        ).read()

        user_prompt = user_template.replace(
            '<perfer>', json.dumps(self.perfer, indent=4, ensure_ascii=False)
        )
        user_prompt = user_prompt.replace(
            '<scenario>', json.dumps(self.scenario, indent=4, ensure_ascii=False)
        )
        user_prompt = user_prompt.replace('<utterance>', self.utterance)
        user_prompt = user_prompt.replace(
            '<tool_result>', json.dumps(self.tool_result, indent=4, ensure_ascii=False)
        )

        return system_prompt, user_prompt

# -------------------------- TOOL EXECUTION --------------------------

def execute_tool_call(tool_call_dict):
    try:
        tool_dir = tool_call_dict["tool_folder_name"]
        func_name = tool_call_dict["function_name"]
        params = tool_call_dict["param"]

        api_path = os.path.join(TOOL_BASE_PATH, tool_dir, 'api.py')
        spec = importlib.util.spec_from_file_location(f"{tool_dir}.api", api_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if tool_dir in CLASS_BASED_TOOLS:
            obj = getattr(module, CLASS_BASED_TOOLS[tool_dir])()
            return True, getattr(obj, func_name)(**params)

        return True, getattr(module, func_name)(**params)

    except Exception as e:
        return False, str(e)

# -------------------------- PIPELINE --------------------------

def process_utterance_pipeline(engine, persona, utterance, tool_defs, history, seq):
    persona_clean = persona['demographic']

    selector = ToolSelectionGeneration(engine, persona_clean, utterance, tool_defs, history)
    first_exec_ok = False
    exec_ok = False
    exec_res = ""

    for attempt in range(3):
        success, call_json = selector.generate(sample_id='0')
        try:
            calls = json.loads(call_json)
            scenario = calls.get("scenario", "unknown")
            tool_call = calls.get("1")
            exec_ok, exec_res = execute_tool_call(tool_call)
            if exec_ok:
                if attempt == 0:
                    first_exec_ok = True
                break
        except:
            continue

    feedback = "failed to extract json"
    fb_ok = False

    for attempt in range(3):
        fb_gen = FeedbackGeneration(engine, persona_clean, utterance, str(exec_res), tool_call)
        fb_ok, feedback_json = fb_gen.generate(sample_id='0')
        try:
            feedback = json.loads(feedback_json).get("feedback", "")
            break
        except:
            continue

    pref_gen = PreferGeneration(
        perfer=persona,
        utterance=utterance,
        scenario=scenario,
        tool=feedback,
        model_name="gpt-4.1",
        save_dir="/tmp"
    )

    pref_ok, pref_json = pref_gen.generate(sample_id="0")
    pref_val = json.loads(pref_json).get("output", "Neutral") if pref_ok else "Neutral"

    pref_map = {
        "Like": "I’m very satisfied with this tool for this scenario.",
        "Dislike": "In this situation, I’m quite annoyed with this tool."
    }

    user_feedback_text = pref_map.get(pref_val, pref_val)

    record = {
        "user_utterance": utterance,
        "agent_reply": str(tool_call) + feedback,
        "user_feedback": user_feedback_text
    }

    final_result = {}
    final_result["1"] = {
        "tool_call": tool_call,
        "tool_raw_result": str(exec_res),
        "generated_feedback": feedback,
        "execute_success": first_exec_ok,
        "3turn_execute_success": exec_ok,
        "feedback_success": fb_ok
    }

    history.append(record)

    return history, {
        "user_utterance": utterance,
        "scenario": scenario,
        "tool_call": final_result,
        "user_feedback": record
    }

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
    save_path = f"{DATA_ROOT}/data/tool_invoc_result/tool_interactions_{model}.json"
    user_info_path = "../profile_based/data/test_set_restructured.json"

    print(f"Loading Qwen3 {model}...")

    engine = PtEngine(
        LLAMA_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    template = get_template(engine.model.model_meta.template, engine.tokenizer)
    engine.default_template = template

    user_data = json.load(open(user_info_path))
    tool_defs = collect_tool_definitions()
    results = {}

    if os.path.exists(save_path):
        results = json.load(open(save_path, 'r')).get("user_interactions", {})

    for u_idx, (user_id, info) in enumerate(user_data.items()):
        if user_id in results and len(results[user_id]) >= amount:
            continue

        print(f"Processing {user_id}")
        history = []
        user_interactions = []

        for i in range(min(amount, len(info["utterance"]))):
            his_new, res = process_utterance_pipeline(
                engine,
                info["persona"],
                info["utterance"][i],
                tool_defs,
                history,
                i + 1
            )
            history = his_new
            user_interactions.append(res)

        results[user_id] = user_interactions

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(
                {"user_interactions": results},
                f,
                ensure_ascii=False,
                indent=2
            )

if __name__ == "__main__":
    main()