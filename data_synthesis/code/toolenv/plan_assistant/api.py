import re
from tkinter import N
from openai import OpenAI

api_key = "sk-xxx" 
class PlanGenerator:
    def __init__(self, api_key= api_key, base_url='', model='gpt-4o'):
        self.api_key = "sk-xxx"
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.system_prompt = (
            "You are an expert consultant tasked with generating professional solution proposals. "
            "When given a specific user need, always output the response strictly following the format below:\n\n"
            "# Solution Proposal\n\n"
            "## 1. Problem Definition\n"
            "- Clear and concise description of the user's need or problem.\n\n"
            "## 2. Objectives\n"
            "- List of goals the solution should achieve.\n\n"
            "## 3. Proposed Solution\n"
            "- Step-by-step actionable plan.\n"
            "- Each step should be specific and practical.\n\n"
            "## 4. Implementation Details\n"
            "- Tools, methods, or resources required.\n"
            "- Estimated timeline or milestones.\n\n"
            "## 5. Risks and Mitigation\n"
            "- Potential challenges and how to address them.\n"
        )

    def generate_plan(self, plan: str, theme: str=None, style: str=None, additional_requirements:str=None, temperature: float = 0):
        
        if theme is None or theme.strip() == "":
            user_prompt = f"Create a detailed {theme} for: {plan} "
        else:
            user_prompt = f"Create a detailed plan for: {plan} "
        if style:   
            user_prompt += f" The style should be {style}."
        if additional_requirements:
            user_prompt += f" Additional requirements: {additional_requirements}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        success = True
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=8192
            )
            output = response.choices[0].message.content
            # Validate output format check
            is_valid = self._check_format(output)
            while not is_valid:
                success = False
                print("Output does not follow the required format.\n")
                output = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=8192
                ).choices[0].message.content
                is_valid = self._check_format(output)
        except Exception as ex:
            success = False
            output = str(ex)
        final_output = {
            "plan": output
        }
        return final_output

    def _check_format(self, text: str) -> bool:
        required_sections = [
            "## 1. Problem Definition",
            "## 2. Objectives",
            "## 3. Proposed Solution",
            "## 4. Implementation Details",
            "## 5. Risks and Mitigation"
        ]
        return all(section in text for section in required_sections)

