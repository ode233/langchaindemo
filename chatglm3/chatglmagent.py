import json
from typing import List, Optional

from chatglm3.chatglm import ChatGLM
from chatglm3.utils import tool_config_from_file


class ChatGLMAgent(ChatGLM):
    max_token: int = 8192
    do_sample: bool = False
    temperature: float = 0.8
    top_p = 0.8
    tokenizer: object = None
    history: List = []
    tool_names: List = []
    has_search: bool = False

    @property
    def _llm_type(self) -> str:
        return "ChatGLM3"


    def _init_history(self, prompt):
        tool_prompts = prompt.split(
            "You have access to the following tools:\n\n")[1].split("\n\nUse a json blob")[0].split("\n")

        tool_names = [tool.split(":")[0] for tool in tool_prompts]
        self.tool_names = tool_names
        tools_json = []
        for i, tool in enumerate(tool_names):
            tool_config = tool_config_from_file(tool)
            if tool_config:
                tools_json.append(tool_config)
            else:
                ValueError(
                    f"Tool {tool} config not found! It's description is {tool_prompts[i]}"
                )

        self.history.append({
            "role": "system",
            "content": "Answer the following questions as best as you can. You have access to the following tools:",
            "tools": tools_json
        })

    def _extract_query(self, prompt: str):
        query = f"""{prompt.split("Human: ")[-1].strip()}"""
        return query

    def _extract_observation(self, prompt: str):
        return_json = prompt.split("Observation: ")[-1].split("\nThought:")[0]
        self.history.append({
            "role": "observation",
            "content": return_json
        })
        return

    def _extract_tool(self):
        if len(self.history[-1]["metadata"]) > 0:
            metadata = self.history[-1]["metadata"]
            content = self.history[-1]["content"]
            if "tool_call" in content:
                for tool in self.tool_names:
                    if tool in metadata:
                        input_para = content.split("='")[-1].split("'")[0]
                        action_json = {
                            "action": tool,
                            "action_input": input_para
                        }
                        self.has_search = True
                        return f"""
Action: 
```
{json.dumps(action_json, ensure_ascii=False)}
```"""
        final_answer_json = {
            "action": "Final Answer",
            "action_input": self.history[-1]["content"]
        }
        self.has_search = False
        return f"""
Action: 
```
{json.dumps(final_answer_json, ensure_ascii=False)}
```"""

    def _call(self, prompt: str, history: List = [], stop: Optional[List[str]] = ["<|user|>"]):
        print("======")
        print(prompt)
        print("======")
        if not self.history:
            self._init_history(prompt)
        if not self.has_search:
            query = self._extract_query(prompt)
        else:
            self._extract_observation(prompt)
            query = ""
        # print("======")
        # print(self.history)
        # print("======")
        super()._call(
            query,
            history=self.history,
            do_sample=self.do_sample,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        response = self._extract_tool()
        history.append((prompt, response))
        return response
