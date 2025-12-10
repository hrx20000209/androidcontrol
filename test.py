import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import json
import tensorflow as tf
from tqdm import tqdm
from android_env.proto.a11y import android_accessibility_forest_pb2


DATA_DIR = "/data/rxhuang/android_control"
OUT_DIR = "/data/rxhuang/android_control_episodes_flat"
os.makedirs(OUT_DIR, exist_ok=True)


def parse_example(raw_bytes: bytes) -> tf.train.Example:
    ex = tf.train.Example()
    ex.ParseFromString(raw_bytes)
    return ex


def save_screenshots(example: tf.train.Example, ep_dir: str) -> None:
    feats = example.features.feature
    imgs = feats["screenshots"].bytes_list.value

    for idx, raw_img in enumerate(imgs):
        out_path = os.path.join(ep_dir, f"screenshot_{idx:03d}.png")
        with open(out_path, "wb") as f:
            f.write(raw_img)


def save_accessibility_trees(example: tf.train.Example, ep_dir: str) -> None:
    feats = example.features.feature
    trees = feats["accessibility_trees"].bytes_list.value

    for idx, raw in enumerate(trees):
        forest = android_accessibility_forest_pb2.AndroidAccessibilityForest()
        forest.ParseFromString(raw)

        # 文本格式保存即可，之后需要的话再解析
        out_path = os.path.join(ep_dir, f"accessibility_tree_{idx:03d}.txt")
        with open(out_path, "w") as f:
            f.write(str(forest))


def save_meta(example: tf.train.Example, ep_dir: str) -> None:
    feats = example.features.feature

    actions = feats["actions"].bytes_list.value
    step_instrs = feats["step_instructions"].bytes_list.value
    goal = feats["goal"].bytes_list.value

    data = {
        "actions": [a.decode("utf-8", errors="ignore") for a in actions],
        "step_instructions": [s.decode("utf-8", errors="ignore") for s in step_instrs],
        "goal": goal[0].decode("utf-8", errors="ignore") if goal else None,
    }

    # 可选：做一个 sanity check
    n_img = len(feats["screenshots"].bytes_list.value)
    n_tree = len(feats["accessibility_trees"].bytes_list.value)
    data["num_screenshots"] = n_img
    data["num_accessibility_trees"] = n_tree

    with open(os.path.join(ep_dir, "meta.json"), "w") as f:
        json.dump(data, f, indent=2)


def main():
    filenames = sorted(tf.io.gfile.glob(os.path.join(DATA_DIR, "android_control-*")))
    print(f"Found {len(filenames)} shards.")

    # 不再预数 total，直接用 tqdm 包一层，不指定 total
    ds = tf.data.TFRecordDataset(filenames, compression_type="GZIP")

    for raw in tqdm(ds):
        raw_bytes = raw.numpy()
        ex = parse_example(raw_bytes)
        feats = ex.features.feature

        ep_id = feats["episode_id"].int64_list.value[0]
        ep_dir = os.path.join(OUT_DIR, f"episode_{ep_id:06d}")
        os.makedirs(ep_dir, exist_ok=True)

        # 保存原始 protobuf
        with open(os.path.join(ep_dir, "example.pb"), "wb") as f:
            f.write(raw_bytes)

        save_screenshots(ex, ep_dir)
        save_accessibility_trees(ex, ep_dir)
        save_meta(ex, ep_dir)

    print("\nFinished!")


if __name__ == "__main__":
    main()
