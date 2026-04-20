import json
import os
import re
import logging
import sys
from typing import Dict, List, Tuple
from pathlib import Path
import base64
import math
import string
import tqdm

sys.path.append("/../data_synthesis/dir_of_llm_generation")
from llm_generation import LLM_Proxy 


# -------------------------- Logging --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/tool_eval.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# -------------------------- Evaluator --------------------------
class ToolCallEvaluator:
    """
    Evaluate tool-call outputs based on persona and user utterances.
    """

    def __init__(self):
        self.prompt_file = "xx/.../eval_prompt.txt"
        self.data_dir = "xx/.../direction_of_tool_interaction_files" 
        self.save_dir = "xx/.../evaluation"

        self.llm = LLM_Proxy()
        self.model_name = "kimi-k2.5" 

        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        """Load evaluation prompt template."""
        with open(self.prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    # -------------------------- Utilities --------------------------

    def _extract_model_name(self, filename: str) -> str:
        match = re.match(r"tool_interactions_(.+)\.json", filename)
        if not match:
            raise ValueError(f"Invalid filename: {filename}")
        return match.group(1)

    def _detect_images(self, tool_call: Dict) -> List[str]:
        """
        Detect image paths in tool output and convert them to base64.
        """
        text = tool_call["1"]
        pattern = r"(/xx/[a-zA-Z0-9\._\-/]+\.jpg)"
        paths = re.findall(pattern, text)

        images = []
        for p in paths:
            path = Path(p)
            if path.exists():
                with open(path, "rb") as f:
                    images.append(base64.b64encode(f.read()).decode("utf-8"))
        return images

    def _distinct_n(self, text: str) -> Tuple[float, float]:
        """
        Compute diversity metrics (D1, D2).
        """
        if not text.strip():
            return 0.0, 0.0

        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        tokens = text.split()

        if not tokens:
            return 0.0, 0.0

        type_count = len(set(tokens))
        token_count = len(tokens)

        d1 = type_count / math.sqrt(2 * token_count)
        d2 = type_count / math.sqrt(token_count)
        return d1, d2

    # -------------------------- Core Evaluation --------------------------

    def single_eval(
        self, persona: Dict, scenario: str, utterance: str, tool_call: Dict
    ) -> Tuple[str, List[int]]:
        """
        Evaluate a single tool-call output.
        Returns:
            explanation (str), score_list (List[int])
        """

        prompt = self.prompt_template.format(
            persona=json.dumps(persona, ensure_ascii=False, indent=2),
            scenario=json.dumps(scenario, ensure_ascii=False, indent=2),
            utterance=json.dumps(utterance, ensure_ascii=False),
            tool_call=json.dumps(tool_call, ensure_ascii=False, indent=2),
        )

        system_prompt = (
            "You are an evaluator assessing a tool-using LLM "
            "in a social support scenario."
        )

        images = self._detect_images(tool_call)

        if images:
            content = [{"type": "text", "text": prompt}]
            for img in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                    }
                )
            _, response = self.llm.llm_request(system_prompt, content, model_name=self.model_name)
        else:
            _, response = self.llm.llm_request(system_prompt, prompt, model_name=self.model_name)

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError("Invalid LLM output format")

        result = json.loads(match.group())

        scores = [max(1, min(5, int(s))) for s in result["score"]]
        explanation = result["explanation"]

        return explanation, scores

    def eval_utterance(
        self, persona: Dict, scenario: str, utterance: str, tool_call: Dict
    ) -> Dict:
        """
        Evaluate one utterance and return structured result.
        """

        explanation, scores = self.single_eval(persona, scenario, utterance, tool_call)
        total_score = sum(scores)

        d1, d2 = self._distinct_n(tool_call["1"])

        return {
            "utterance": utterance,
            "scenario": scenario,
            "tool_call": tool_call,
            "score": {
                "explanation": explanation,
                "list": scores,
                "total": total_score,
            },
            "distinct": {"d1": d1, "d2": d2},
        }

    # -------------------------- Data Loading --------------------------

    def load_data(self, path: str) -> Dict:
        """Load tool-call JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "user_interactions" not in data:
            raise ValueError("Missing 'user_interactions'")
        return data["user_interactions"]

    # -------------------------- Main Loop --------------------------

    def run(self, user_info_path: str):
        """
        Main evaluation loop over all models.
        """

        with open(user_info_path, "r", encoding="utf-8") as f:
            user_info = json.load(f)

        results = {"models": {}}

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if not file.startswith("tool_interactions_"):
                    continue

                model = self._extract_model_name(file)
                path = os.path.join(root, file)

                logger.info(f"Evaluating model: {model}")

                data = self.load_data(path)

                model_scores = []

                for user_id, interactions in tqdm.tqdm(data.items(), desc=model):
                    if user_id not in user_info:
                        continue

                    persona = user_info[user_id]["persona"]
                    scenarios = user_info[user_id]["scenarios"]
                    utterances = user_info[user_id]["utterance"]

                    for i, interaction in enumerate(interactions):
                        if "tool_call_all" in interaction:
                            tool_call = {"1": interaction["tool_call_all"]}
                        elif "1" in interaction:
                            tool_call = {"1": interaction["1"]["generated_feedback"]}
                        else:
                            tool_call = {"1": interaction["tool_call"]["1"]["generated_feedback"]}

                        result = self.eval_utterance(
                            persona,
                            scenarios[i],
                            utterances[i],
                            tool_call,
                        )

                        model_scores.append(result["score"]["total"])

                avg_score = sum(model_scores) / len(model_scores) if model_scores else 0

                results["models"][model] = {
                    "avg_score": avg_score,
                    "count": len(model_scores),
                }

        save_path = os.path.join(self.save_dir, "final_results.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved results to {save_path}")


# -------------------------- Entry --------------------------

if __name__ == "__main__":
    evaluator = ToolCallEvaluator()
    evaluator.run("/xx/data/test_set_restructured.json")