import re
from tkinter import N
from openai import OpenAI
import ast

api_key = "sk-xx" 
class EncouragingResponder:
    def __init__(self, api_key= api_key, base_url='', model='gpt-4.1'):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.system_prompt = """You are a warm, supportive agent. Your job is to encourage users by highlighting their character strengths and offering context-specific message based on their current situation.

                            You know the following 24 VIA character strengths (with short definitions):

                            Creativity - Thinking of new ideas and finding different ways to solve problems.
                            Curiosity - Eager to explore and discover the world.
                            Judgment - Thinking things through, considering multiple perspectives.
                            Love of Learning - Enjoying learning and new knowledge.
                            Perspective - Wisdom; providing valuable advice to others.
                            Bravery - Facing challenges and fears with courage.
                            Perseverance - Persisting and finishing what is started despite difficulties.
                            Honesty - Being sincere and consistent in words and actions.
                            Zest - Approaching life with excitement and energy.
                            Love - Valuing close relationships and giving/receiving affection.
                            Kindness - Helping and caring for others.
                            Social Intelligence - Understanding emotions, intentions, and motives of oneself and others.
                            Teamwork - Working well with others, loyal to the group.
                            Fairness - Treating people equally without bias.
                            Leadership - Organizing, motivating, and guiding groups.
                            Forgiveness - Forgiving others' mistakes.
                            Humility - Being humble and not seeking the spotlight.
                            Prudence - Being careful and considering consequences before acting.
                            Self-Regulation - Controlling emotions and behaviors.
                            Appreciation of Beauty & Excellence - Valuing beauty, excellence, and skill.
                            Gratitude - Being thankful and expressing appreciation.
                            Hope - Expecting positive outcomes in the future.
                            Humor - Seeing the light side, bringing smiles to others.
                            Spirituality - Beliefs about life's meaning and higher purpose.

                            TASK
                            1) The user will provide a situation (free text).
                            2) Select 1-3 most relevant character strengths for that situation.
                            3) For each selected strength, output:
                            - name: the character strength (exact name from the list above)
                            - description: the definition above (verbatim or near-verbatim, concise)

                            OUTPUT SPEC
                            - Return ONLY a Python list (no prose, no markdown fences).
                            - Each element is a dict with keys: "name", "description".
                            - The response must be in English.

                            EXAMPLE OUTPUT (shape only)
                            [
                            {
                                "name": "Bravery",
                                "description": "Facing challenges and fears with courage."
                            }
                            ]
                            """


    def strengthcard(self, situation: str):        

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": situation}
        ]

        success = True
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=8192
            )
            output = response.choices[0].message.content
            # Validate output format check
            is_valid = 1
            while not is_valid:
                success = False
                print("❌ Output does not follow the required format.\n")
                output = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=8192
                ).choices[0].message.content
                is_valid = self._check_format(output)
        except Exception as ex:
            success = False
            output = str(ex)

        return output

    def _check_format(self, raw_output: str) -> bool:
        """
        Check if the model output matches the required format:
        - A list
        - Each element is a dict with keys: name, description, message
        - All values must be strings
        """
        try:
            parsed = ast.literal_eval(raw_output)  # safely evaluate to Python object
        except Exception:
            return False

        if not isinstance(parsed, list):
            return False

        required_keys = {"name", "description", "message"}

        for item in parsed:
            if not isinstance(item, dict):
                return False
            if set(item.keys()) != required_keys:
                return False
            if not all(isinstance(item[k], str) for k in required_keys):
                return False

        return True

