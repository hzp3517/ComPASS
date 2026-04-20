from tkinter import N
from zhipuai import ZhipuAI
import sys
from pathlib import Path 
from APIclient import onechatAPIclient


class RoleplayAgent:
    def __init__(self, api_key="xxx"):
        """
        初始化角色扮演代理
        """
        self.client = ZhipuAI(api_key=api_key)
        self.api_client = onechatAPIclient()

    def build_prompt(self, name=None,ip= None,relationship= None,style=None,user_name= None,user_input=None) -> list:
        
        role_info = (
            f"You are roleplaying as {name} from {ip}.\n"
            f"Your relationship with the user is: {relationship}.\n"
            f"the user is named {user_name}.\n"
            f"Please respond in a {style} style, staying in character."
        )

        return [
            {"role": "user", "content": role_info},
            {"role": "user", "content": f'The user says: "{user_input}"'}
        ]

    def roleplay_agent(self, name=None,ip= None,relationship= None,style=None,user_name= None, user_input=None) -> str:
     
        messages = self.build_prompt(name,ip,relationship,style,user_name,user_input)
        response = self.client.chat.completions.create(
            model="charglm-4",
            messages=messages
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    api_key = "xxx"
    agent = RoleplayAgent()

    user_input = "people can never understand each other, your attemption to seek comminication is a waste of time. i dont think people care to listen to you."
    reply = agent.roleplay_agent(
        name="xxx",
        ip= None,
        relationship= None,
        style=None,
        user_name= None,
        user_input=user_input)
    print(reply)
