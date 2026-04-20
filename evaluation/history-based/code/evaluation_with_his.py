import json
import os
import re
import logging
import sys
from typing import Dict, List
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
        logging.FileHandler('/xxx/logs/history_tool_eval_qwen3-max.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

sys.path.append("/xxx/data_synthesis_v2/code/")
from llm_generation import LLM_Proxy

# -------------------------- Core Evaluation Class --------------------------
class ToolCallEvaluator:
    """
    Tool Call Result Evaluator
    Scores model-generated tool calls based on user persona and utterances
    Metrics: empathy, helpfulness, preference alignment, fluency, safety, information richness
    Supports image detection & multi-modal evaluation
    """
    def __init__(self):
        self.single_score_eval_prompt_file = "/xxx/evaluation/profile_based/prompt/single_score_prompt.txt"
        self.tool_call_base_dir = "/xxx/evaluation/history_based/data/tool_invoc_result"
        self.evaluation_save_base_dir = "/xxx/evaluation/history_based/data/eval"
        self.eval_result_save_path = os.path.join(self.evaluation_save_base_dir, "tool_call_evaluation_results.json")

        # LLM Evaluator Initialization
        self.evaluator_llm = LLM_Proxy()
        self.eval_model = "kimi-k2.5"
        self.eval_prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load and validate evaluation prompt template"""
        if not os.path.exists(self.single_score_eval_prompt_file):
            raise FileNotFoundError(f"Evaluation prompt file not found: {self.single_score_eval_prompt_file}")

        with open(self.single_score_eval_prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        logger.info("Successfully loaded evaluation prompt template")
        return prompt

    def _extract_model_name(self, file_name: str) -> str:
        """Extract model name from result filename"""
        pattern = r"tool_interactions_(.+)\.json"
        match = re.match(pattern, file_name)
        if match:
            return match.group(1)
        raise ValueError(f"Invalid filename format: {file_name}")

    def single_score_eval(self, persona: dict, scenario, utterance: str, tool_call: dict) -> list:
        """
        Core evaluation for a single tool call result
        Returns: explanation string, score list [0-5] * 6
        """
        filled_prompt = self.eval_prompt_template.format(
            persona=json.dumps(persona, indent=4, ensure_ascii=False),
            utterance=json.dumps(utterance, ensure_ascii=False),
            scenario=json.dumps(scenario, indent=4, ensure_ascii=False),
            tool_call=json.dumps(tool_call, indent=4, ensure_ascii=False)
        )

        system_prompt = "you are a helpful user simulator assistance. you will judge the performance of a tool use LLM's social support ability based on user persona"
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
            logger.warning("REGENERATING RESPONSE - JSON FORMAT ERROR")
            filled_prompt += """
                Please strictly output ONLY the JSON string in the required format:
                {"explanation":"detailed analysis for each metric","score":[0-5,0-5,0-5,0-5,0-5,0-5]}
                No extra text, punctuation, or explanations outside the JSON!
            """
            image_num, image_base64_list = self.image_detect(tool_call)
            prompt = [{"type": "text", "text": filled_prompt}]

            if image_num > 0:
                for img_b64 in image_base64_list:
                    prompt.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })
                success, llm_response = self.evaluator_llm.llm_request(system_prompt, prompt, model_name=self.eval_model)
            else:
                success, llm_response = self.evaluator_llm.llm_request(system_prompt, filled_prompt, model_name=self.eval_model)

            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            retry_cnt += 1

        if json_match:
            eval_result = json.loads(json_match.group())

            if "score" not in eval_result or "explanation" not in eval_result:
                raise ValueError("JSON missing 'score' or 'explanation' field")

            score_list = eval_result["score"]
            explanation = eval_result["explanation"]
            if not isinstance(score_list, list) or len(score_list) != 6:
                raise ValueError(f"Score list invalid: expected 6 integers, got {score_list}")

            score_list = [max(1, min(5, int(s))) for s in score_list]
            logger.info(f"Successfully extracted scores: {score_list}")
            return explanation, score_list
        else:
            raise ValueError(f"Failed to extract JSON from LLM response: {llm_response[:100]}")

    def image_detect(self, tool_interaction):
        """Detect and encode images from tool result paths"""
        image_base64_list = []
        try:
            tool_result = tool_interaction["1"]
            pattern = r'(/xxx[a-zA-Z0-9\._\-/]+\.jpg)' # you have to change this to your own image direction
            all_matches = re.findall(pattern, tool_result, re.IGNORECASE)

            if all_matches:
                for img_str in all_matches:
                    image_path = Path(img_str)
                    resolved_path = image_path.resolve(strict=False)
                    if resolved_path.exists():
                        with open(resolved_path, "rb") as img_file:
                            image_bytes = img_file.read()
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                            image_base64_list.append(image_base64)
                    else:
                        logger.warning(f"Image path not found: {resolved_path}")

            return len(image_base64_list), image_base64_list

        except Exception as e:
            logger.error(f"Error in image_detect: {str(e)}")
            return 0, []

    def eval_single_utterance(self, persona: Dict, scenario: str, utterance: str, tool_call: Dict) -> Dict:
        """Evaluate single user utterance with full metadata"""
        try:
            explanation, basic_score = self.single_score_eval(persona, scenario, utterance, tool_call)
            total_score = sum(basic_score)
            text = tool_call["1"]
            d1, d2 = self.distinct_n(text)

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
        """Calculate Distinct-1 and Distinct-2 diversity scores"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0, 0.0

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
            raise FileNotFoundError(f"Tool call file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "user_interactions" not in data:
            raise ValueError("Missing core field 'user_interactions' in tool call file")
        logger.info(f"Successfully loaded tool call file: {file_path}")
        return data["user_interactions"]

    def main_eval(self, user_info_path: str) -> None:
        """Main evaluation pipeline: evaluate all models and users"""
        # Load user profile data
        with open(user_info_path, 'r', encoding='utf-8') as f:
            user_info_data = json.load(f)
        logger.info(f"Successfully loaded user profile file: {user_info_path}")

        final_eval_results = {"model_evaluations": {}}
        file_list = []

        # Collect target model files
        print("models in queue:")
        for root, dirs, files in os.walk(self.tool_call_base_dir):
            for file in files:
                if file.startswith("tool_interactions_") and file.endswith(".json"):
                    model_name = self._extract_model_name(file)
                    if model_name in ['compass']:
                        print(model_name)
                        file_list.append((root, file, model_name))

        # Evaluate each model
        for root, file, model_name in tqdm.tqdm(file_list, desc="model_file", unit="file"):
            try:
                print(f"\nevaluating model:{model_name}")
                print("*" * 10)
                file_path = os.path.join(root, file)
                logger.info(f"Start evaluating model: {model_name} (file: {file_path})")

                tool_call_data = self.load_tool_call_data(file_path)

                model_eval_result = {
                    "user_evaluations": {},
                    "overall_avg_score": None,
                    "valid_utterance_count": 0,
                    "execute_succ_rate": 0
                }

                utt_num = 0

                # User processing loop
                user_iter = tqdm.tqdm(list(tool_call_data.items()), desc=f"[{model_name}]users", unit="user", leave=False)

                for user_id, user_interactions in user_iter:
                    print(user_id)
                    user_iter.set_description(f"[{model_name}] Processing user {user_id}")

                    if user_id not in user_info_data:
                        logger.warning(f"User {user_id} not found in profile data, skipping")
                        continue

                    user_persona = user_info_data[user_id]["persona"]
                    user_scenario = user_info_data[user_id].get("scenarios")
                    user_utterance = user_info_data[user_id].get("utterance")

                    user_utterance_evals = []
                    user_valid_scores = []

                    # Process utterances from index 12
                    for utterance_idx, interaction in enumerate(user_interactions[12:]):
                        utterance = user_utterance[utterance_idx + 12]
                        utt_num += 1

                        if "tool_call_failed" in interaction:
                            tool_call_pos = {"1": "Tool call failed"}
                        else:
                            tool_call_pos = {"1": interaction["tool_call"]["1"]["generated_feedback"]}

                        scenario = user_scenario[utterance_idx + 12]

                        if not utterance:
                            logger.warning(f"User {user_id} utterance {utterance_idx} is empty, skipping")
                            continue

                        # Run evaluation
                        utterance_eval = self.eval_single_utterance(
                            persona=user_persona,
                            scenario=scenario,
                            utterance=utterance,
                            tool_call=tool_call_pos
                        )

                        utterance_eval["user_id"] = user_id
                        utterance_eval["utterance_index"] = utterance_idx + 12
                        user_utterance_evals.append(utterance_eval)
                        logger.info(f"Processed: {utterance_eval}")

                    model_eval_result["user_evaluations"][user_id] = user_utterance_evals

                    # Calculate user average score
                    if user_valid_scores:
                        user_avg_score = sum(user_valid_scores) / len(user_valid_scores)
                        model_eval_result[f"user_{user_id}_avg_score"] = user_avg_score
                        logger.info(f"User {user_id} average score: {user_avg_score:.2f}")
                    else:
                        model_eval_result[f"user_{user_id}_avg_score"] = None

                # Save model results
                model_save_file = f"/xxx/evaluation/history_based/data/kimi_eval/{model_name}v2_eval.json"
                with open(model_save_file, "w", encoding='utf-8') as f:
                    json.dump(model_eval_result, f, ensure_ascii=False, indent=2)

                final_eval_results["model_evaluations"][model_name] = model_eval_result

            except Exception as e:
                logger.error(f"Model evaluation failed for {file}: {str(e)}", exc_info=True)
                continue

# -------------------------- Main Entry --------------------------
if __name__ == "__main__":
    USER_INFO_PATH = "/xxx/evaluation/profile_based/data/test_set_restructured.json"

    try:
        evaluator = ToolCallEvaluator()
        evaluator.main_eval(user_info_path=USER_INFO_PATH)
    except Exception as e:
        logger.critical(f"Evaluation pipeline failed: {str(e)}", exc_info=True)
        print(f"\nEvaluation pipeline failed: {str(e)}")