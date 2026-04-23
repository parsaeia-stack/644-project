import argparse
import json
from collections import Counter
from pathlib import Path


SLOTS = ["pickup", "dropoff", "ridetype", "time", "passengers", "payment"]
INVALID_LABELS = {
    "invalid_action",
    "invalid_request_already_confirmed_slot",
    "invalid_confirm_missing_slot",
    "invalid_confirm_already_confirmed_slot",
}
QUESTIONABLE_LABELS = {"questionable_reask_filled_unconfirmed_slot"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate saved test dialogues against slot-filling state rules."
    )
    parser.add_argument("--path", default="test_results.json", help="Path to test_results.json")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum invalid/questionable examples to print per policy.",
    )
    return parser.parse_args()


def empty_state():
    state = {}
    for slot in SLOTS:
        state[f"{slot}_filled"] = 0
        state[f"{slot}_conf"] = 0
    return state


def split_action(action):
    if action.startswith("request_confirm_"):
        return "confirm", action.removeprefix("request_confirm_")
    if action.startswith("request_"):
        return "request", action.removeprefix("request_")
    return "unknown", action


def update_state(state, response):
    state = dict(state)
    for slot in SLOTS:
        if response in (f"provide_{slot}", f"provide_wrong_{slot}"):
            state[f"{slot}_filled"] = 1
            state[f"{slot}_conf"] = 0
            return state
        if response == f"confirm_pos_{slot}":
            state[f"{slot}_conf"] = 1
            return state
        if response == f"confirm_neg_{slot}":
            state[f"{slot}_filled"] = 0
            state[f"{slot}_conf"] = 0
            return state
    return state


def classify_action(action, state):
    kind, slot = split_action(action)
    if slot not in SLOTS:
        return "invalid_action"

    filled = state[f"{slot}_filled"]
    confirmed = state[f"{slot}_conf"]

    if kind == "request":
        if filled == 0 and confirmed == 0:
            return "valid_request_missing_slot"
        if filled == 1 and confirmed == 0:
            return "questionable_reask_filled_unconfirmed_slot"
        return "invalid_request_already_confirmed_slot"

    if kind == "confirm":
        if filled == 1 and confirmed == 0:
            return "valid_confirm_filled_unconfirmed_slot"
        if filled == 0:
            return "invalid_confirm_missing_slot"
        return "invalid_confirm_already_confirmed_slot"

    return "invalid_action"


def validate_policy(policy_name, episodes, max_examples):
    counts = Counter()
    invalid_examples = []
    questionable_examples = []
    success_count = 0
    total_turns = 0

    for episode in episodes:
        success_count += bool(episode["success"])
        state = empty_state()

        for turn in episode["dialogue"]:
            label = classify_action(turn["action"], state)
            counts[label] += 1
            total_turns += 1

            example = {
                "episode": episode["episode"],
                "turn": turn["turn"],
                "action": turn["action"],
                "user_response": turn["user_response"],
                "state_before": dict(state),
                "label": label,
            }

            if label in INVALID_LABELS and len(invalid_examples) < max_examples:
                invalid_examples.append(example)
            if label in QUESTIONABLE_LABELS and len(questionable_examples) < max_examples:
                questionable_examples.append(example)

            state = update_state(state, turn["user_response"])

    invalid_total = sum(counts[label] for label in INVALID_LABELS)
    questionable_total = sum(counts[label] for label in QUESTIONABLE_LABELS)
    valid_total = total_turns - invalid_total - questionable_total

    print(f"\n{policy_name}")
    print("-" * len(policy_name))
    print(f"Episodes: {len(episodes)}")
    print(f"Successes: {success_count}")
    print(f"Failures/timeouts: {len(episodes) - success_count}")
    print(f"Turns checked: {total_turns}")
    print(f"Valid turns: {valid_total}")
    print(f"Questionable re-asks: {questionable_total}")
    print(f"Invalid turns: {invalid_total}")
    print(f"Valid request missing slot: {counts['valid_request_missing_slot']}")
    print(f"Valid confirm filled/unconfirmed slot: {counts['valid_confirm_filled_unconfirmed_slot']}")

    if invalid_examples:
        print("\nInvalid examples:")
        for example in invalid_examples:
            print(json.dumps(example, indent=2))

    if questionable_examples:
        print("\nQuestionable examples:")
        for example in questionable_examples:
            print(json.dumps(example, indent=2))

    return {
        "episodes": len(episodes),
        "turns": total_turns,
        "valid": valid_total,
        "questionable": questionable_total,
        "invalid": invalid_total,
    }


def main():
    args = parse_args()
    path = Path(args.path)
    data = json.loads(path.read_text())

    print(f"Validating {path.resolve()}")
    print("Rules:")
    print("- request_<slot> is valid when the slot is missing")
    print("- request_confirm_<slot> is valid when the slot is filled but unconfirmed")
    print("- confirm_neg_<slot> resets the slot to missing")
    print("- confirm_pos_<slot> marks the slot confirmed")

    totals = Counter()
    for policy_name, episodes in data.items():
        summary = validate_policy(policy_name, episodes, args.max_examples)
        totals.update(summary)

    print("\nOverall")
    print("-------")
    print(f"Episodes: {totals['episodes']}")
    print(f"Turns checked: {totals['turns']}")
    print(f"Valid turns: {totals['valid']}")
    print(f"Questionable re-asks: {totals['questionable']}")
    print(f"Invalid turns: {totals['invalid']}")

    if totals["invalid"] == 0:
        print("\nResult: PASS - all dialogues obey the slot-filling state rules.")
    else:
        print("\nResult: FAIL - at least one dialogue contains an invalid system action.")


if __name__ == "__main__":
    main()
