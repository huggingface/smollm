from .base import SmolTool
from typing import Generator, List, Dict, Any, Callable
import json
import re
from datetime import datetime
import random
from transformers import tool, CodeAgent
import requests
import webbrowser

@tool
def get_random_number_between(min: int, max: int) -> str:
    """
    Gets a random number between min and max.

    Args:
        min: The minimum number.
        max: The maximum number.

    Returns:
        A random number between min and max.
    """
    return f"{random.randint(min, max):d}"


@tool
def get_weather(city: str) -> str:
    """
    Returns the weather forecast for a given city.

    Args:
        city: The name of the city.

    Returns:
        A string with a mock weather forecast.
    """
    url = 'https://wttr.in/{}?format=+%C,+%t'.format(city)
    res = requests.get(url).text

    return f"The weather in {city} is {res.split(',')[0]} with a high of {res.split(',')[1][:-2]} degrees Celsius."

@tool
def get_current_time() -> str:
    """
    This is a tool that returns the current time.
    It returns the current time as HH:MM.
    """
    return f"The current time is {datetime.now().hour}:{datetime.now().minute}."

@tool
def open_webbrowser(url: str) -> str:
    """
    This is a tool that opens a web browser to the given website.
    If the user asks to open a website or a browser, you should use this tool.

    Args:
        url: The url to open.
    """
    webbrowser.open(url)
    return f"I opened {url.replace('https://', '').replace('www.', '')} in the browser."


class SmolToolAgent(SmolTool):
    def __init__(self):
        self.tools = [get_random_number_between, get_current_time, open_webbrowser, get_weather]
        self.toolbox = {tool.name: tool for tool in self.tools}
        self.json_code_agent = CodeAgent(tools=self.tools, llm_engine=self.llm_engine, system_prompt=self._get_system_prompt())
        super().__init__(
            model_repo="andito/SmolLM2-1.7B-Instruct-F16-GGUF",
            model_filename="smollm2-1.7b-8k-dpo-f16.gguf",
            system_prompt=self._get_system_prompt(),
            prefix_text=""
        )

    def llm_engine(self, messages, stop_sequences=["Task", "<|endoftext|>"]) -> str:
        output = ""
        for chunk in self.model.create_chat_completion(
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
            top_p=1.0,
            top_k=50,
            repeat_penalty=1.0,
            stream=True
        ):
            content = chunk['choices'][0]['delta'].get('content')
            if content:
                if content in ["<end_action>", "<|endoftext|>"]:
                    break
                output += content
        return output

    def _get_system_prompt(self) -> str:
        return """You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out and refuse to answer.
If the given question lacks the parameters required by the function, also point it out.

You have access to the following tools:
<<tool_descriptions>>

<<managed_agents_descriptions>>

You can use imports in your code, but only from the following list of modules: <<authorized_imports>>

The output MUST strictly adhere to the following format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.
<tool_call>[
{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
... (more tool calls as required)
]</tool_call>"""

    def _parse_response(self, text: str) -> List[Dict[str, Any]]:
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return json.loads(matches[0])
        return text

    def _call_tools(self, tool_calls: List[Dict[str, Any]]) -> List[str]:
        tool_responses = []
        for tool_call in tool_calls:
            if tool_call["name"] in self.toolbox:
                tool_responses.append(
                    self.toolbox[tool_call["name"]](**tool_call["arguments"])
                )
            else:
                tool_responses.append(f"Tool {tool_call['name']} not found.")
        return tool_responses

    def process(self, text: str) -> Generator[str, None, None]:
        response = self.json_code_agent.run(text, return_generated_code=True)
        # Parse and execute the tool calls
        try:
            tool_calls = self._parse_response(response)
            if tool_calls in [response, [], ""]:
                yield response
                return
            tool_responses = self._call_tools(tool_calls)
        except Exception as e:
            print("error", e)
            yield response
            return

        # Yield each tool response
        for response in tool_responses:
            yield str(response)