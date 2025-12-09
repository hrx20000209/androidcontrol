"""
Agent implementations for AITW evaluation using HuggingFace models.
"""

from typing import Dict, Any, Optional
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor


PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""



class VLMAgent:
    """
    Agent that uses vision-language model with screenshot images.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize VLMAgent with HuggingFace VLM.
        
        Args:
            model_name: Name of the HuggingFace VLM to use (e.g., "Qwen/Qwen2-VL-7B-Instruct")
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading VLMAgent model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print(f"VLMAgent loaded on {self.device}")
    
    def _build_prompt(self, task: str) -> str:
        return PROMPT.format(instruction=task, language="English")