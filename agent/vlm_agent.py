"""
VLM Agent implementations for AITW evaluation using HuggingFace models.
"""

from typing import Dict, Any, Optional
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor


PROMPT = """You are a GUI agent. You are given a task and your action history, with current screenshot. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```

## Action History
{history}


## Action Space

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- The `Action` must strictly follow the action space format.
- Choose the best next action to move toward the goal.

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
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print(f"VLMAgent loaded on {self.device}")
    
    def _build_prompt(self, task, history):
        return PROMPT.format(history=history, instruction=task, language="English")
    
    def run_step(self, task, screenshot, history) -> str:
        """
        Execute one step of the agent.

        Args:
            - task: Task description string
            - screenshot: PIL Image
            - history: List of previous actions taken
                
        Returns:
            Raw model output string
        """
        prompt = self._build_prompt(task, history)
        
        # Prepare messages with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[screenshot],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate response
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
        response = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        return response
