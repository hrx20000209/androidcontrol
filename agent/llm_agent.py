"""
LLM Agent implementations for AITW evaluation using HuggingFace models.
"""

from typing import Dict, Any, Optional
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor


PROMPT = """You are a GUI agent. You are given a task and your action history, with XML tree representations of the UI. You need to perform the next action to complete the task. 
## Output Format
"""

class XMLAgent:
    """
    Agent that uses XML tree representation for UI understanding with LLM.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize XMLAgent with HuggingFace LLM.
        
        Args:
            model_name: Name of the HuggingFace model to use (e.g., "Qwen/Qwen2.5-7B-Instruct")
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading XMLAgent model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print(f"XMLAgent loaded on {self.device}")
    
    def _build_prompt(self, task: str, xml: str) -> str:
        return PROMPT
    
    def run_step(self, state: Dict[str, Any]) -> str:
        """
        Execute one step of the agent.
        
        Args:
            state: Dictionary containing:
                - task: Task description string
                - xml: XML tree string
                - image: PIL Image (not used by XMLAgent)
                - episode_id: Episode identifier
                - step_id: Step identifier
                
        Returns:
            Raw model output string
        """
        prompt = self._build_prompt(state['task'], state['xml'])
        
        # Tokenize and generate
        messages = [{"role": "user", "content": prompt}]
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
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
        
        # Decode only the generated tokens (excluding input)
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response
