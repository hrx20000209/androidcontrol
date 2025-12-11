import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json
from torch.utils.data import DataLoader
from data.android_control_dataset import AndroidControlEpisodeDataset
from data.xml_tree import convert_forest_any_to_xml
from agent.llm_agent import XMLAgent
from actions.action_matching import match_actions
from actions.action_parser import pred_action_parser, gt_action_parser


##########################################################
#  Episode Evaluation
##########################################################
def evaluate_episode(agent, loader, log_f):
    global_type_correct = 0
    global_full_correct = 0
    global_total_steps = 0

    global_per_type_stats = {}

    # === 全局错误计数（统一类型） ===
    global_errors = 0

    for ep_data in loader:
        ep_id = ep_data["episode_id"]
        num_steps = len(ep_data["actions"])
        history = []

        print(f"\n===== Episode {ep_id} =====")
        log_f.write(f"\n===== Episode {ep_id} =====\n")
        log_f.write(f"Steps in this episode: {num_steps}\n")
        
        if ep_id > 400:
            break

        for step_id in range(num_steps):
            global_total_steps += 1

            # ---- GT ----
            gt_raw = ep_data["actions"][step_id]
            try:
                gt = gt_action_parser(gt_raw)
            except Exception as e:
                global_errors += 1
                log_f.write(f"[ERROR][gt_parse] {e}\n")
                gt = {"action_type": "wait"}
            gt_type = gt["action_type"]

            xml = convert_forest_any_to_xml(ep_data["accessibility_trees"][step_id])

            if gt_type not in global_per_type_stats:
                global_per_type_stats[gt_type] = {
                    "total": 0,
                    "type_correct": 0,
                    "full_correct": 0,
                }
            global_per_type_stats[gt_type]["total"] += 1

            # ---- Pred ----
            try:
                pred_raw = agent.run_step(
                    task=ep_data["goal"],
                    instruction=ep_data["step_instructions"][step_id],
                    xml=xml,
                    history=history,
                )
            except Exception as e:
                global_errors += 1
                log_f.write(f"[ERROR][agent] {e}\n")
                pred_raw = "Action: wait()"

            print(pred_raw)

            try:
                pred = pred_action_parser(pred_raw)
            except Exception as e:
                global_errors += 1
                log_f.write(f"[ERROR][pred_parse] {e}\n")
                pred = {"action_type": "wait"}

            history.append(pred)

            # ---- Matching ----
            try:
                type_ok, full_ok = match_actions(gt, pred)
            except Exception as e:
                global_errors += 1
                log_f.write(f"[ERROR][match] {e}\n")
                type_ok, full_ok = False, False

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
        print(f"Cumulative Errors: {global_errors}")

        log_f.write(
            f"\nCumulative Overall Type Accuracy: "
            f"{overall_type_acc:.2f}% ({global_type_correct}/{global_total_steps})\n"
        )
        log_f.write(
            f"Cumulative Overall Full Accuracy: "
            f"{overall_full_acc:.2f}% ({global_full_correct}/{global_total_steps})\n"
        )
        log_f.write(f"Cumulative Errors: {global_errors}\n")

        print("\n------------------------------------------\n")
        log_f.write("\n------------------------------------------\n")

    print("\n============== ERROR SUMMARY ==============")
    print(f"Total Errors: {global_errors}")
    print("==========================================")

    log_f.write("\n============== ERROR SUMMARY ==============\n")
    log_f.write(f"Total Errors: {global_errors}\n")
    log_f.write("==========================================\n")


##########################################################
# main
##########################################################
def main():
    dataset = AndroidControlEpisodeDataset("/data/rxhuang/android_control_episodes_flat")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    log_f = open("./logs/llm_agent.log", "w")
    agent = XMLAgent(model_name="./models/qwen3_1.7B")  # or LLMAgent(...)

    evaluate_episode(agent, loader, log_f)

    log_f.close()
    print("\nSaved evaluation log → logs/android_control_agent_eval.log")


if __name__ == "__main__":
    main()
