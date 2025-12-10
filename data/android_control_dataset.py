import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class AndroidControlEpisodeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # 所有 episode_* 目录
        self.episodes = sorted([
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if d.startswith("episode_") and os.path.isdir(os.path.join(root_dir, d))
        ])

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep_dir = self.episodes[idx]
        ep_id = int(ep_dir.split("_")[-1])

        # ---------- load meta.json ----------
        with open(os.path.join(ep_dir, "meta.json"), "r") as f:
            meta = json.load(f)

        actions = meta["actions"]
        step_instructions = meta["step_instructions"]
        goal = meta["goal"]
        num_steps = len(step_instructions)

        # ---------- load screenshots ----------
        screenshots = []
        for i in range(meta["num_screenshots"]):
            img_path = os.path.join(ep_dir, f"screenshot_{i:03d}.png")
            img = Image.open(img_path).convert("RGB")
            screenshots.append(img)

        # ---------- load accessibility trees ----------
        accessibility_trees = []
        for i in range(meta["num_accessibility_trees"]):
            tree_path = os.path.join(ep_dir, f"accessibility_tree_{i:03d}.txt")
            with open(tree_path, "r") as f:
                accessibility_trees.append(f.read())

        return {
            "episode_id": ep_id,
            "screenshots": screenshots,
            "accessibility_trees": accessibility_trees,
            "actions": actions,
            "step_instructions": step_instructions,
            "goal": goal,
            "num_steps": num_steps,
        }
