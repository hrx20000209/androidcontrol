import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
from torch.utils.data import DataLoader
from data.android_control_dataset import AndroidControlEpisodeDataset
from agent.vlm_agent import VLMAgent
from actions.action_matching import match_actions
from actions.action_parser import pred_action_parser, gt_action_parser


##########################################################
#  Episode Evaluation
##########################################################
import traceback

##########################################################
#  Episode Evaluation
##########################################################
def evaluate_episode(agent, loader, log_f):
    global_type_correct = 0
    global_full_correct = 0
    global_total_steps = 0

    global_per_type_stats = {}

    # 统一的 error 计数
    global_error_count = 0

    for ep_data in loader:
        ep_id = ep_data["episode_id"]
        num_steps = len(ep_data["actions"])
        history = []
        if ep_id > 400:
            break

        print(f"\n===== Episode {ep_id} =====")
        log_f.write(f"\n===== Episode {ep_id} =====\n")
        log_f.write(f"Steps in this episode: {num_steps}\n")

        for step_id in range(num_steps):

            try:
                global_total_steps += 1

                # ---------------- GT ----------------
                gt_raw = ep_data["actions"][step_id]
                gt = gt_action_parser(gt_raw)
                gt_type = gt["action_type"]

                if gt_type not in global_per_type_stats:
                    global_per_type_stats[gt_type] = {
                        "total": 0,
                        "type_correct": 0,
                        "full_correct": 0,
                    }
                global_per_type_stats[gt_type]["total"] += 1

                # ---------------- Pred ----------------
                pred_raw = agent.run_step(
                    task=ep_data["goal"],
                    instruction=ep_data["step_instructions"][step_id],
                    screenshot=ep_data["screenshots"][step_id],
                    history=history,
                )
                print(f"Raw model output: {pred_raw}")
                pred = pred_action_parser(pred_raw)
                # 修复坐标：模型输出的是缩小版坐标，需要乘回去
                if pred["action_type"] in ["click", "long_press"]:
                    pred["x"] *= agent.resolution_scale
                    pred["y"] *= agent.resolution_scale
                history.append(pred)

                # ---------------- Match ----------------
                type_ok, full_ok = match_actions(gt, pred)

                if type_ok:
                    global_type_correct += 1
                    global_per_type_stats[gt_type]["type_correct"] += 1

                if full_ok:
                    global_full_correct += 1
                    global_per_type_stats[gt_type]["full_correct"] += 1

                print(f"\nStep {step_id}:")
                print(f"  GT:   {gt}")
                print(f"  Pred: {pred}")
                print(f"  type_ok={type_ok}, full_ok={full_ok}")

                log_f.write(f"\nStep {step_id}:\n")
                log_f.write(f"  GT:   {gt}\n")
                log_f.write(f"  Pred: {pred}\n")
                log_f.write(f"  type_ok={type_ok}, full_ok={full_ok}\n")

            except Exception as e:
                # ======================
                # 捕获所有错误并继续执行
                # ======================
                global_error_count += 1

                print(f"\n[ERROR] Step {step_id} failed. Skipping this step.")
                log_f.write(f"\n[ERROR] Step {step_id} failed. Skipping this step.\n")

                # 打印堆栈（可注释掉）
                traceback.print_exc()
                log_f.write(traceback.format_exc() + "\n")

                # 该 step 记为 wait 动作（防止 history 断裂）
                history.append({"action_type": "wait"})

                continue

        # ======================================================
        # Episode-level cumulative accuracy
        # ======================================================
        print("\n--- Cumulative Per-Type Accuracy (up to this episode) ---")
        log_f.write("\n--- Cumulative Per-Type Accuracy (up to this episode) ---\n")

        for t, d in global_per_type_stats.items():
            tot = d["total"]
            type_c = d["type_correct"]
            full_c = d["full_correct"]

            type_acc = type_c / tot * 100.0
            full_acc = full_c / tot * 100.0

            line = (f"{t}: type_acc={type_acc:.2f}% ({type_c}/{tot}), "
                    f"full_acc={full_acc:.2f}% ({full_c}/{tot})")
            print(line)
            log_f.write(line + "\n")

        overall_type_acc = global_type_correct / global_total_steps * 100.0
        overall_full_acc = global_full_correct / global_total_steps * 100.0

        print(f"\nCumulative Overall Type Accuracy: {overall_type_acc:.2f}% "
              f"({global_type_correct}/{global_total_steps})")
        print(f"Cumulative Overall Full Accuracy: {overall_full_acc:.2f}% "
              f"({global_full_correct}/{global_total_steps})")
        print(f"Total Errors so far: {global_error_count}")

        log_f.write(
            f"\nCumulative Overall Type Accuracy: "
            f"{overall_type_acc:.2f}% ({global_type_correct}/{global_total_steps})\n"
        )
        log_f.write(
            f"Cumulative Overall Full Accuracy: "
            f"{overall_full_acc:.2f}% ({global_full_correct}/{global_total_steps})\n"
        )
        log_f.write(
            f"Total Errors so far: {global_error_count}\n"
        )

        print("\n------------------------------------------\n")
        log_f.write("\n------------------------------------------\n")


##########################################################
# main
##########################################################
def main():
    dataset = AndroidControlEpisodeDataset("/data/rxhuang/android_control_episodes_flat")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    log_f = open("./logs/vlm_agent_7B_1_4.log", "w")
    agent = VLMAgent(model_name="./models/ui_tars_7B", resolution_scale=4)  # or LLMAgent(...)

    evaluate_episode(agent, loader, log_f)

    log_f.close()
    print("\nSaved evaluation log → logs/android_control_agent_eval.log")


if __name__ == "__main__":
    main()
