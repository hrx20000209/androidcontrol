"""
LLM Agent implementations for AITW evaluation using HuggingFace models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT = """You are a GUI agent.
You MUST NOT output any thinking, explanation, reasoning, or chain-of-thought.
You MUST NOT output <think>, Thought:, analysis, or any hidden reasoning.

You must output ONLY one line:
Action: ...

## Action Space
click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='')
open_app(app_name=\'\')
scroll(point='<point>x1 y1</point>', direction='down or up or left or right')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_home()
press_back()
finished(content='xxx')

## Action History:
{history}

## User Instruction
{instruction}

## Current UI (XML):
{xml}

## Action History
{history}

"""


class XMLAgent:
    """
    Agent that uses XML tree representation for UI understanding with a language-only LLM.
    """

    def __init__(self, model_name: str):
        """
        Initialize XMLAgent with HuggingFace LLM.

        Args:
            model_name: HuggingFace model name, e.g. "Qwen/Qwen2.5-7B-Instruct"
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading XMLAgent model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"XMLAgent loaded on {self.device}")

    def _build_prompt(self, task, instruction, xml, history):
        """
        Format the prompt exactly like the vision agent.
        """
        return PROMPT.format(
            history=history,
            xml=xml,
            task=task,
            instruction=instruction
        )

    def run_step(self, task: str, instruction: str, xml: str, history: str) -> str:
        """
        Execute one step of the agent using XML + text-only LLM.

        Args:
            task: task description
            xml: UI structure as XML string
            history: previous actions in text format

        Returns:
            Raw LLM output (string)
        """
        prompt = self._build_prompt(task, instruction, xml, history)

        # Chat template
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,              # 为了使用top_p
                temperature=0.8,            # 近似0，不会报错
                top_p=0.8,
                repetition_penalty=1.2,
            )

        # Only decode new tokens
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        return response
