# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import re
import ast
import math
import json


IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def convert_point_to_coordinates(text, is_answer=False):
    # 匹配 <bbox> 后面的四个数字
    pattern = r"<point>(\d+)\s+(\d+)</point>"

    def replace_match(match):
        x1, y1 = map(int, match.groups())
        x = (x1 + x1) // 2  # 使用截断取整
        y = (y1 + y1) // 2  # 使用截断取整
        if is_answer:
            return f"({x},{y})"  # 只返回 (x, y) 格式
        return f"({x},{y})"  # 返回带标签的格式

    # 去掉 [EOS] 并替换 <bbox> 坐标
    text = re.sub(r"\[EOS\]", "", text)
    return re.sub(pattern, replace_match, text).strip()


# 定义一个函数来解析每个 action
def parse_action(action_str):
    try:
        # 解析字符串为 AST 节点
        node = ast.parse(action_str, mode='eval')

        # 确保节点是一个表达式
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        # 获取表达式的主体
        call = node.body

        # 确保主体是一个函数调用
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # 获取函数名
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        # 获取关键字参数
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            # 处理不同类型的值，这里假设都是常量
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # 兼容旧版本 Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {'function': func_name, 'args': kwargs}

    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None


def escape_single_quotes(text):
    # 匹配未转义的单引号（不匹配 \\'）
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def linear_resize(height: int,
                  width: int,
                  factor: int = IMAGE_FACTOR,
                  min_pixels: int = MIN_PIXELS,
                  max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    if width * height > max_pixels:
        """
        如果图片超过/低于像素限制，则计算一个缩放因子resize_factor，使图片的像素数缩小到等于或小于max_pixels。这个缩放因子是通过开平方根计算的，确保纵横比保持不变,这样原始的相对坐标可以不经转换直接复用
        """
        resize_factor = math.sqrt(max_pixels / (width * height))
        width, height = int(width * resize_factor), int(height * resize_factor)
    if width * height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (width * height))
        width, height = math.ceil(width * resize_factor), math.ceil(
            height * resize_factor)

    return height, width


def smart_resize(height: int,
                 width: int,
                 factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def parse_action_to_structure_output(text):
    """
    输入模型输出格式：
        Thought: ...
        Action: click(point="(120,300)")

    输出结构：
        {
            "thought": "...",
            "action_type": "click",
            "action_inputs": {"x":120, "y":300},
            "text": 原始文本
        }
    """

    text = text.strip()
    text = text.replace("[EOS]", "")

    # -----------------------------
    # 1) 提取 Thought
    # -----------------------------
    thought = None
    m = re.search(r"Thought\s*:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
    if m:
        thought = m.group(1).strip()

    # -----------------------------
    # 2) 提取 Action 字符串
    # -----------------------------
    if "Action:" not in text:
        raise ValueError("No Action: found")

    action_str = text.split("Action:", 1)[1].strip()

    # 截断到第一个 ')'
    if ")" in action_str:
        action_str = action_str.split(")", 1)[0] + ")"

    # -----------------------------
    # 3) 使用 parse_action 解析
    # -----------------------------
    parsed = parse_action(action_str)
    if parsed is None:
        raise ValueError(f"Failed to parse: {action_str}")

    atype = parsed["function"]
    params = parsed["args"]

    # -----------------------------
    # 4) 转换成 Android Control 所需结构
    # -----------------------------
    action_inputs = {}

    # point="(x,y)"
    if "point" in params and params["point"]:
        xy_str = params["point"].replace("(", "").replace(")", "")
        x, y = map(float, xy_str.split(","))
        action_inputs["x"] = x
        action_inputs["y"] = y

    # 文本内容
    if "content" in params:
        action_inputs["content"] = params["content"]

    # scroll 参数
    if "direction" in params:
        action_inputs["direction"] = params["direction"]

    # open_app 参数
    if "app_name" in params:
        action_inputs["app_name"] = params["app_name"]

    return [{
        "thought": thought,
        "action_type": atype,
        "action_inputs": action_inputs,
        "text": text,
    }]



def parsing_response_to_pyautogui_code(responses,
                                       image_height: int,
                                       image_width: int,
                                       input_swap: bool = True) -> str:
    '''
    将M模型的输出解析为OSWorld中的action，生成pyautogui代码字符串
    参数:
        response: 包含模型输出的字典，结构类似于：
        {
            "action_type": "hotkey",
            "action_inputs": {
                "hotkey": "v ctrl",
                "start_box": None,
                "end_box": None
            }
        }
    返回:
        生成的pyautogui代码字符串
    '''

    pyautogui_code = f"import pyautogui\nimport time\n"
    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):
        if "observation" in response:
            observation = response["observation"]
        else:
            observation = ""

        if "thought" in response:
            thought = response["thought"]
        else:
            thought = ""

        if response_id == 0:
            pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
        else:
            pyautogui_code += f"\ntime.sleep(1)\n"

        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})

        if action_type == "hotkey":
            # Parsing hotkey action
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"

            elif hotkey == "arrowup":
                hotkey = "up"

            elif hotkey == "arrowdown":
                hotkey = "down"

            if hotkey:
                # Handle other hotkeys
                keys = hotkey.split()  # Split the keys by space
                convert_keys = []
                for key in keys:
                    if key == "space":
                        key = ' '
                    convert_keys.append(key)
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})"

        elif action_type in ["press", "keydown"]:
            # Parsing press action
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            if key_to_press == "arrowleft":
                key_to_press = "left"

            elif key_to_press == "arrowright":
                key_to_press = "right"

            elif key_to_press == "arrowup":
                key_to_press = "up"

            elif key_to_press == "arrowdown":
                key_to_press = "down"

            elif key_to_press == "space":
                key_to_press = " "

            if key_to_press:
                # Simulate pressing a single key
                pyautogui_code += f"\npyautogui.keyDown({repr(key_to_press)})"

        elif action_type in ["release", "keyup"]:
            # Parsing press action
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            if key_to_press == "arrowleft":
                key_to_press = "left"

            elif key_to_press == "arrowright":
                key_to_press = "right"

            elif key_to_press == "arrowup":
                key_to_press = "up"

            elif key_to_press == "arrowdown":
                key_to_press = "down"

            elif key_to_press == "space":
                key_to_press = " "

            if key_to_press:
                # Simulate pressing a single key
                pyautogui_code += f"\npyautogui.keyUp({repr(key_to_press)})"

        elif action_type == "type":
            # Parsing typing action using clipboard
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            stripped_content = content
            if content.endswith("\n") or content.endswith("\\n"):
                stripped_content = stripped_content.rstrip("\\n").rstrip("\n")
            if content:
                if input_swap:
                    pyautogui_code += f"\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{stripped_content}')"
                    pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"
                else:
                    pyautogui_code += f"\npyautogui.write('{stripped_content}', interval=0.1)"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"

        elif action_type in ["drag", "select"]:
            # Parsing drag or select action based on start and end_boxes
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                x1, y1, x2, y2 = eval(
                    start_box)  # Assuming box is in [x1, y1, x2, y2]
                sx = round(float((x1 + x2) / 2) * image_width, 3)
                sy = round(float((y1 + y2) / 2) * image_height, 3)
                x1, y1, x2, y2 = eval(
                    end_box)  # Assuming box is in [x1, y1, x2, y2]
                ex = round(float((x1 + x2) / 2) * image_width, 3)
                ey = round(float((y1 + y2) / 2) * image_height, 3)
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})\n"
                    f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n")

        elif action_type == "scroll":
            # Parsing scroll action
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, x2, y2 = eval(
                    start_box)  # Assuming box is in [x1, y1, x2, y2]
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)

                # # 先点对应区域，再滚动
                # pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
            else:
                x = None
                y = None
            direction = action_inputs.get("direction", "")

            if x == None:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5)"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5)"
            else:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

        elif action_type in [
                "click", "left_single", "left_double", "right_single", "hover"
        ]:
            # Parsing mouse click actions
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            if start_box:
                start_box = eval(start_box)
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2 = x1
                    y2 = y1
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"

        elif action_type in ["finished"]:
            pyautogui_code = f"DONE"

        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code


def add_box_token(input_string):
    # Step 1: Split the string into individual actions
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            # Step 2: Extract coordinates (start_box or end_box) using regex
            coordinates = re.findall(
                r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)

            updated_action = action  # Start with the original action
            for coord_type, x, y in coordinates:
                # Convert x and y to integers
                updated_action = updated_action.replace(
                    f"{coord_type}='({x},{y})'",
                    f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'")
            processed_actions.append(updated_action)

        # Step 5: Reconstruct the final string
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string


def gt_action_parser(gt):
    """
    统一解析 Ground Truth，支持 JSON 字符串 或 dict。
    
    返回 canonical 结构：
        - click / long_press: {"action_type": "click", "touch": (y,x), "lift": (y,x)}
        - scroll: {"action_type": "scroll", "direction": ...}
        - open_app: {"action_type": "open_app", "app_name": ...}
        - input_text: {"action_type":"input_text","text":...}
        - navigate_home / navigate_back / wait
    """

    # -------- ① 如果是字符串，先解析成 dict --------
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except Exception as e:
            raise ValueError(f"Failed to parse GT JSON: {gt}") from e

    if not isinstance(gt, dict):
        raise ValueError(f"GT is not a dict: {gt}")

    atype = gt.get("action_type")
    if atype is None:
        raise ValueError(f"No action_type in GT: {gt}")

    # =============== CLICK / LONG PRESS ===============
    if atype in ["click", "long_press"]:
        x = float(gt["x"])
        y = float(gt["y"])
        return {
            "action_type": atype,
            "x": x,
            "y": y,
        }

    # =============== SCROLL ===============
    if atype == "scroll":
        return {
            "action_type": "scroll",
            "direction": gt.get("direction", "down")
        }

    # =============== OPEN APP ===============
    if atype == "open_app":
        return {
            "action_type": "open_app",
            "app_name": gt.get("app_name", "")
        }

    # =============== TEXT INPUT ===============
    if atype == "input_text":
        return {
            "action_type": "input_text",
            "text": gt.get("text", "")
        }

    # =============== HOME / BACK ===============
    if atype == "navigate_home":
        return {"action_type": "navigate_home"}

    if atype == "navigate_back":
        return {"action_type": "navigate_back"}

    # =============== WAIT ===============
    if atype == "wait":
        return {"action_type": "wait"}

    raise ValueError(f"Unsupported GT action type: {atype}")


def pred_action_parser(model_output):
    """
    Robust version.
    """

    text = (model_output or "").strip().replace("[EOS]", "")

    if "Action:" not in text:
        return {"action_type": "wait"}

    act_str = text.split("Action:", 1)[1].strip()
    act_str = act_str.split("\n", 1)[0].strip()

    # ===== 自动容错修复 =====

    # 补齐右括号
    if act_str.count("(") > act_str.count(")"):
        act_str += ")"

    # 补齐双引号
    if act_str.count("\"") % 2 == 1:
        act_str += "\""

    # 补齐单引号
    if act_str.count("'") % 2 == 1:
        act_str += "'"

    # 修复 click(start_box='(635,759)
    act_str = re.sub(
        r"\((\d+),\s*(\d+)\)['\"]?$",
        r"(\1,\2))",
        act_str
    )

    # 修复 <point>300 500</point>
    act_str = re.sub(
        r"<point>(\d+)\s+(\d+)</point>",
        lambda m: f"({m.group(1)},{m.group(2)})",
        act_str
    )

    # 截到最后一个 ')'
    if ")" in act_str:
        act_str = act_str[:act_str.rfind(")") + 1]

    # ===== 调 parse_action =====
    try:
        parsed = parse_action(act_str)
    except Exception:
        parsed = None

    if parsed is None:
        # 尝试仅解析坐标
        coords = re.findall(r"\d+", act_str)
        if len(coords) >= 2:
            try:
                x, y = map(float, coords[:2])
                return {"action_type": "click", "x": x, "y": y}
            except:
                pass

        return {"action_type": "wait"}

    atype = parsed["function"]
    args = parsed["args"]

    # ===== CLICK / LONG PRESS =====
    if atype in ["click", "long_press"]:
        # 1) point="(x,y)"
        if "point" in args and args["point"]:
            xy = args["point"].replace("(", "").replace(")", "")
            x, y = map(float, xy.split(","))
            return {
                "action_type": atype,
                "x": x,
                "y": y,
            }

        # 2) start_box 当成坐标中心点
        if "start_box" in args and args["start_box"]:
            xy = args["start_box"].replace("(", "").replace(")", "")
            x, y = map(float, xy.split(","))
            return {
                "action_type": atype,
                "x": x,
                "y": y,
            }
    # ===== SCROLL =====
    if atype == "scroll":
        direction = args.get("direction", "").lower().strip()
        if direction not in ["up", "down", "left", "right"]:
            direction = "down"
        return {"action_type": "scroll", "direction": direction}

    # ===== OPEN APP =====
    if atype == "open_app":
        return {
            "action_type": "open_app",
            "app_name": args.get("app_name", "")
        }

    # ===== TEXT INPUT =====
    if atype == "type":
        return {
            "action_type": "input_text",
            "text": args.get("content", "")
        }

    # ===== SYSTEM KEYS =====
    if atype == "press_home":
        return {"action_type": "navigate_home"}

    if atype == "press_back":
        return {"action_type": "navigate_back"}

    return {"action_type": "wait"}
