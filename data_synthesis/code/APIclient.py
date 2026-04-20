from openai import OpenAI

api_key = "sk-xx"

class onechatAPIclient:
    def __init__(self, api_key=api_key, base_url='...', model='gpt-4.1'):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def apicall(self, system_prompt=None, user_prompt=None):

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        output = response.choices[0].message.content
        return output

if __name__ == "__main__":
    client = onechatAPIclient()
    print(client.apicall(system_prompt="you are a helpfull assistance",user_prompt="can you search this link?'https://www.mentalhealth.org.uk/a-to-z/l/loneliness'"))
