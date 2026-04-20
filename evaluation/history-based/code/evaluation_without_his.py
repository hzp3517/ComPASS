import json
import os
import re
import logging
import sys
from typing import Dict, List
from datetime import datetime
import tqdm
import base64
from pathlib import Path
import math
import string

# -------------------------- Logging Configuration --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler('/xxx/logs/history_tool_eval.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

sys.path.append("/xxx/data_synthesis/code/")
from llm_generation import LLM_Proxy

# -------------------------- Core Evaluation Class --------------------------
class ToolCallEvaluator:
    """
    Tool Call Result Evaluator
    Scores model-generated tool calls based on user persona and utterances
    Evaluation metrics: empathy, helpfulness, preference alignment, fluency, safety, information richness
    """
    def __init__(self):
        self.single_score_eval_prompt_file = "/xxx/evaluation/profile_based/prompt/single_score_prompt.txt"
        self.tool_call_base_dir = "/xxx/evaluation/history_based/data/tool_invoc_result"
        self.evaluation_save_base_dir = "/xxx/evaluation/history_based/data/eval"
        self.eval_result_save_path = os.path.join(self.evaluation_save_base_dir, "tool_call_evaluation_results.json")

        # LLM Evaluator Setup
        self.evaluator_llm = LLM_Proxy()
        self.eval_model = "kimi-k2.5"
        self.eval_prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load and validate evaluation prompt template"""
        if not os.path.exists(self.single_score_eval_prompt_file):
            raise FileNotFoundError(f"Evaluation prompt file missing: {self.single_score_eval_prompt_file}")

        with open(self.single_score_eval_prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        logger.info("Successfully loaded evaluation prompt template")
        return prompt

    def _extract_model_name(self, file_name: str) -> str:
        """Extract model name from result file name"""
        pattern = r"tool_interactions_without_his_(.+)\.json"
        match = re.match(pattern, file_name)
        if match:
            return match.group(1)
        raise ValueError(f"Invalid file name format: {file_name}")

    def single_score_eval(self, persona: dict, scenario, utterance: str, tool_call: dict) -> List:
        """
        Evaluate a single tool call result
        Returns: explanation string, score list [0-5] * 6
        """
        filled_prompt = self.eval_prompt_template.format(
            persona=json.dumps(persona, indent=4, ensure_ascii=False),
            utterance=json.dumps(utterance, ensure_ascii=False),
            scenario=json.dumps(scenario, indent=4, ensure_ascii=False),
            tool_call=json.dumps(tool_call, indent=4, ensure_ascii=False)
        )

        system_prompt = "You are a helpful user simulator assistant. Judge the LLM's tool-use social support ability based on user persona."
        image_num, image_base64_list = self.image_detect(tool_call)
        prompt = [{"type": "text", "text": filled_prompt}]

        # Add images if detected
        if image_num > 0:
            for img_b64 in image_base64_list:
                prompt.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            success, llm_response = self.evaluator_llm.llm_request(system_prompt, prompt, model_name=self.eval_model)
        else:
            success, llm_response = self.evaluator_llm.llm_request(system_prompt, filled_prompt, model_name=self.eval_model)

        logger.debug(f"LLM evaluation response: {llm_response[:100]}...")

        # Retry logic for JSON extraction
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        retry_cnt = 0
        while not json_match and retry_cnt < 3:
            logger.warning("Regenerating response - JSON format error")
            filled_prompt += """
                Please output ONLY the JSON string in this format:
                {"explanation":"detailed analysis","score":[0-5,0-5,0-5,0-5,0-5,0-5]}
                No extra text outside the JSON object.
            """
            if image_num > 0:
                success, llm_response = self.evaluator_llm.llm_request(system_prompt, prompt, model_name=self.eval_model)
            else:
                success, llm_response = self.evaluator_llm.llm_request(system_prompt, filled_prompt, model_name=self.eval_model)
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            retry_cnt += 1

        if not json_match:
            raise ValueError(f"Failed to extract valid JSON from LLM response: {llm_response[:100]}")

        # Parse and validate result
        eval_result = json.loads(json_match.group())
        if "score" not in eval_result or "explanation" not in eval_result:
            raise ValueError("Missing 'score' or 'explanation' fields in JSON")

        score_list = eval_result["score"]
        if not isinstance(score_list, list) or len(score_list) != 6:
            raise ValueError(f"Score list must contain 6 integers, got: {score_list}")

        # Clamp scores to valid range
        score_list = [max(1, min(5, int(s))) for s in score_list]
        logger.info(f"Successfully extracted scores: {score_list}")
        return eval_result["explanation"], score_list

    def image_detect(self, tool_interaction) -> tuple[int, List]:
        """Detect and encode images from tool result paths"""
        image_base64_list = []
        try:
            tool_result = tool_interaction["1"]
            pattern = r'(/xxx[a-zA-Z0-9\._\-/]+\.jpg)' # you have to change this to your own image direction
            matches = re.findall(pattern, tool_result, re.IGNORECASE)

            for img_path_str in matches:
                img_path = Path(img_path_str)
                if img_path.exists():
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode('utf-8')
                        image_base64_list.append(b64)
                        logger.info(f"Loaded image: {img_path}")
                else:
                    logger.warning(f"Image path not found: {img_path}")
        except Exception as e:
            logger.error(f"Image detection error: {str(e)}")

        return len(image_base64_list), image_base64_list

    def eval_single_utterance(self, persona: Dict, scenario: str, utterance: str, tool_call: Dict) -> Dict:
        """Evaluate single user utterance and tool call"""
        try:
            explanation, basic_score = self.single_score_eval(persona, scenario, utterance, tool_call)
            total_score = sum(basic_score)
            d1, d2 = self.distinct_n(tool_call["1"])

            return {
                "utterance": utterance,
                "scenario": scenario,
                "tool_call": tool_call,
                "basic_score": {
                    "explanation": explanation,
                    "score_list": basic_score,
                    "total_score": total_score
                },
                "distinct_n": {
                    "D_1": d1,
                    "D_2": d2
                },
                "error": None
            }
        except Exception as e:
            logger.error(f"Single utterance evaluation failed: {str(e)}", exc_info=True)
            return {
                "utterance": utterance,
                "scenario": scenario,
                "tool_call": tool_call,
                "error": str(e),
                "basic_score": None
            }

    def distinct_n(self, text):
        """Calculate distinct-1 and distinct-2 scores"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0, 0.0

        # Clean and tokenize text
        text = text.lower()
        clean_text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = clean_text.split()
        token_count = len(tokens)

        if token_count == 0:
            return 0.0, 0.0

        type_count = len(set(tokens))
        d1 = type_count / math.sqrt(2 * token_count)
        d2 = type_count / math.sqrt(token_count)
        return d1, d2

    def load_tool_call_data(self, file_path: str) -> Dict:
        """Load and validate tool call result file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tool call file missing: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "user_interactions" not in data:
            raise ValueError("Missing core field 'user_interactions' in tool call file")
        logger.info(f"Successfully loaded tool call data: {file_path}")
        return data["user_interactions"]

    def main_eval(self, user_info_path: str) -> None:
        """Main evaluation pipeline: evaluate all models and users"""
        # Load user profile data
        with open(user_info_path, 'r', encoding='utf-8') as f:
            user_info_data = json.load(f)
        logger.info(f"Successfully loaded user profile data: {user_info_path}")

        final_eval_results = {"model_evaluations": {}}
        file_list = []

        # Collect target evaluation files
        logger.info("Models in evaluation queue:")
        for root, _, files in os.walk(self.tool_call_base_dir):
            for file in files:
                if file.startswith("tool_interactions_without_his_") and file.endswith(".json"):
                    model_name = self._extract_model_name(file)
                    if model_name in ['gpt-5.1']:
                        logger.info(model_name)
                        file_list.append((root, file, model_name))

        # Evaluate each model
        for root, file, model_name in tqdm.tqdm(file_list, desc="Processing models", unit="file"):
            model_save_path = f"/xxx/evaluation/history_based/data/kimi_eval/{model_name}v2_nohis_eval.json"


            logger.info(f"\nStarting evaluation for model: {model_name}")
            file_path = os.path.join(root, file)
            tool_call_data = self.load_tool_call_data(file_path)

            model_eval_result = {
                "user_evaluations": {},
                "overall_avg_score": None,
                "valid_utterance_count": 0,
                "execute_succ_rate": 0.0,
                "3turn_execute_succ_rate": 0.0
            }

            all_valid_scores = []
            total_utterances = 0

            # Process each user
            user_progress = tqdm.tqdm(tool_call_data.items(), desc=f"[{model_name}] Processing users", unit="user", leave=False)
            for user_id, interactions in user_progress:
                user_progress.set_description(f"[{model_name}] Processing user {user_id}")

                if user_id not in user_info_data:
                    logger.warning(f"User {user_id} not found in profile data, skipping")
                    continue

                user_persona = user_info_data[user_id]["persona"]
                user_scenarios = user_info_data[user_id].get("scenarios")
                user_utterances = user_info_data[user_id]["utterance"]
                user_eval_list = []
                user_valid_scores = []

                # Process each utterance
                for idx, interaction in enumerate(interactions):
                    utt_idx = idx + 12
                    utterance = user_utterances[utt_idx]
                    total_utterances += 1

                    tool_call_data_pos = {"1": interaction["tool_call"]["1"]["generated_feedback"]}
                    scenario = user_scenarios[utt_idx]

                    # Evaluate single turn
                    eval_result = self.eval_single_utterance(
                        persona=user_persona,
                        scenario=scenario,
                        utterance=utterance,
                        tool_call=tool_call_data_pos
                    )

                    eval_result["user_id"] = user_id
                    eval_result["utterance_index"] = utt_idx
                    user_eval_list.append(eval_result)

                    # Collect valid scores
                    if eval_result["basic_score"] and eval_result["basic_score"]["total_score"] is not None:
                        score = eval_result["basic_score"]["total_score"]
                        user_valid_scores.append(score)
                        all_valid_scores.append(score)

                model_eval_result["user_evaluations"][user_id] = user_eval_list

                # Calculate user average score
                if user_valid_scores:
                    avg = sum(user_valid_scores) / len(user_valid_scores)
                    model_eval_result[f"user_{user_id}_avg_score"] = avg
                    logger.info(f"User {user_id} average score: {avg:.2f}")

            # Calculate model-level metrics
            if all_valid_scores:
                model_eval_result["overall_avg_score"] = sum(all_valid_scores) / len(all_valid_scores)
                model_eval_result["valid_utterance_count"] = len(all_valid_scores)

            # Save model-specific results
            try:
                with open(model_save_path, "w", encoding='utf-8') as f:
                    json.dump(model_eval_result, f, ensure_ascii=False, indent=2)
                logger.info(f"Successfully saved results for {model_name}")
            except Exception as e:
                logger.error(f"Failed to save results for {model_name}: {str(e)}")

            final_eval_results["model_evaluations"][model_name] = model_eval_result

        logger.info("Evaluation pipeline completed successfully")

# -------------------------- Main Execution --------------------------
if __name__ == "__main__":
    USER_INFO_PATH = "/xxx/evaluation/profile_based/data/test_set_restructured.json"

    try:
        evaluator = ToolCallEvaluator()
        evaluator.main_eval(user_info_path=USER_INFO_PATH)
    except Exception as e:
        logger.critical(f"Evaluation pipeline failed: {str(e)}", exc_info=True)
        print(f"Evaluation pipeline failed: {str(e)}")