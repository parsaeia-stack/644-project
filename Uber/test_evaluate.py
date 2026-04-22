import json
import random
from stable_baselines3 import PPO
from environment import UberDialogueEnv
from test_simulated_user import SimulatedUser, SLOT_VALUES
import numpy as np

ACTIONS = [
    "request_pickup", "request_dropoff", "request_ridetype", "request_time",
    "request_passengers", "request_payment",
    "request_confirm_pickup", "request_confirm_dropoff", "request_confirm_ridetype",
    "request_confirm_time", "request_confirm_passengers", "request_confirm_payment"
]

# Generate fixed test set of 1000 user goals
random.seed(99)  # fixed seed so test set is always the same
TEST_SET = []
for _ in range(1000):
    TEST_SET.append({
        "pickup":     random.choice(SLOT_VALUES["pickup"]),
        "dropoff":    random.choice(SLOT_VALUES["dropoff"]),
        "ridetype":   random.choice(SLOT_VALUES["ridetype"]),
        "time":       random.choice(SLOT_VALUES["time"]),
        "passengers": random.choice(SLOT_VALUES["passengers"]),
        "payment":    random.choice(SLOT_VALUES["payment"])
    })


def run_episode(model, goal, episode_seed=0):
    """Run one episode with a fixed goal and test simulated user."""
    
    
    user = SimulatedUser()
    user.goal = goal.copy()
    
    
    state = {
        "pickup_filled": 0, "pickup_conf": 0,
        "dropoff_filled": 0, "dropoff_conf": 0,
        "ridetype_filled": 0, "ridetype_conf": 0,
        "time_filled": 0, "time_conf": 0,
        "passengers_filled": 0, "passengers_conf": 0,
        "payment_filled": 0, "payment_conf": 0,
    }
    current_values = {
        "pickup": None, "dropoff": None,
        "ridetype": None, "time": None,
        "passengers": None, "payment": None
    }
    
    
    max_turns = 50
    dialogue = []
    
    for turn in range(1, max_turns + 1):
        # Get observation
        obs = np.array([
            state["pickup_filled"], state["pickup_conf"],
            state["dropoff_filled"], state["dropoff_conf"],
            state["ridetype_filled"], state["ridetype_conf"],
            state["time_filled"], state["time_conf"],
            state["passengers_filled"], state["passengers_conf"],
            state["payment_filled"], state["payment_conf"],
        ], dtype=np.int8)
        
        # Policy picks action (no exploration)
        action, _ = model.predict(obs, deterministic=True)
        action_name = ACTIONS[action]
        
        
        user_response = user.respond(action_name, current_values)
        
       
        _update_state(state, current_values, user_response, user)
        
        dialogue.append({
            "turn": turn,
            "action": action_name,
            "user_response": user_response,
            "state": {
                "pickup_filled": int(state["pickup_filled"]),
                "pickup_conf": int(state["pickup_conf"]),
                "dropoff_filled": int(state["dropoff_filled"]),
                "dropoff_conf": int(state["dropoff_conf"]),
                "ridetype_filled": int(state["ridetype_filled"]),
                "ridetype_conf": int(state["ridetype_conf"]),
                "time_filled": int(state["time_filled"]),
                "time_conf": int(state["time_conf"]),
                "passengers_filled": int(state["passengers_filled"]),
                "passengers_conf": int(state["passengers_conf"]),
                "payment_filled": int(state["payment_filled"]),
                "payment_conf": int(state["payment_conf"])
                }
        })
        
        # Check if done
        if _is_done(state):
            return {"success": True, "turns": turn, "dialogue": dialogue}
    
    return {"success": False, "turns": max_turns, "dialogue": dialogue}


def _update_state(state, current_values, user_response, user):
    if user_response == "provide_pickup":
        current_values["pickup"] = user.goal["pickup"]
        state["pickup_filled"] = 1
        state["pickup_conf"] = 0
    elif user_response == "provide_wrong_pickup":
        current_values["pickup"] = user._wrong_value("pickup")
        state["pickup_filled"] = 1
        state["pickup_conf"] = 0
    elif user_response == "provide_dropoff":
        current_values["dropoff"] = user.goal["dropoff"]
        state["dropoff_filled"] = 1
        state["dropoff_conf"] = 0
    elif user_response == "provide_wrong_dropoff":
        current_values["dropoff"] = user._wrong_value("dropoff")
        state["dropoff_filled"] = 1
        state["dropoff_conf"] = 0
    elif user_response == "provide_ridetype":
        current_values["ridetype"] = user.goal["ridetype"]
        state["ridetype_filled"] = 1
        state["ridetype_conf"] = 0
    elif user_response == "provide_wrong_ridetype":
        current_values["ridetype"] = user._wrong_value("ridetype")
        state["ridetype_filled"] = 1
        state["ridetype_conf"] = 0
    elif user_response == "provide_time":
        current_values["time"] = user.goal["time"]
        state["time_filled"] = 1
        state["time_conf"] = 0
    elif user_response == "provide_wrong_time":
        current_values["time"] = user._wrong_value("time")
        state["time_filled"] = 1
        state["time_conf"] = 0
    elif user_response == "provide_passengers":
        current_values["passengers"] = user.goal["passengers"]
        state["passengers_filled"] = 1
        state["passengers_conf"] = 0
    elif user_response == "provide_wrong_passengers":
        current_values["passengers"] = user._wrong_value("passengers")
        state["passengers_filled"] = 1
        state["passengers_conf"] = 0
    elif user_response == "provide_payment":
        current_values["payment"] = user.goal["payment"]
        state["payment_filled"] = 1
        state["payment_conf"] = 0
    elif user_response == "provide_wrong_payment":
        current_values["payment"] = user._wrong_value("payment")
        state["payment_filled"] = 1
        state["payment_conf"] = 0
    elif user_response == "confirm_pos_pickup":
        state["pickup_conf"] = 1
    elif user_response == "confirm_pos_dropoff":
        state["dropoff_conf"] = 1
    elif user_response == "confirm_pos_ridetype":
        state["ridetype_conf"] = 1
    elif user_response == "confirm_pos_time":
        state["time_conf"] = 1
    elif user_response == "confirm_pos_passengers":
        state["passengers_conf"] = 1
    elif user_response == "confirm_pos_payment":
        state["payment_conf"] = 1
    elif user_response == "confirm_neg_pickup":
        state["pickup_filled"] = 0
        state["pickup_conf"] = 0
        current_values["pickup"] = None
    elif user_response == "confirm_neg_dropoff":
        state["dropoff_filled"] = 0
        state["dropoff_conf"] = 0
        current_values["dropoff"] = None
    elif user_response == "confirm_neg_ridetype":
        state["ridetype_filled"] = 0
        state["ridetype_conf"] = 0
        current_values["ridetype"] = None
    elif user_response == "confirm_neg_time":
        state["time_filled"] = 0
        state["time_conf"] = 0
        current_values["time"] = None
    elif user_response == "confirm_neg_passengers":
        state["passengers_filled"] = 0
        state["passengers_conf"] = 0
        current_values["passengers"] = None
    elif user_response == "confirm_neg_payment":
        state["payment_filled"] = 0
        state["payment_conf"] = 0
        current_values["payment"] = None


def _is_done(state):
    return all(state[k] == 1 for k in state)


def evaluate_policy(model, policy_name):
    results = []
    random.seed(42)  # fixed seed for reproducibility

    for i, goal in enumerate(TEST_SET):
        result = run_episode(model, goal, episode_seed=i)
        result["goal"] = goal
        result["episode"] = i
        results.append(result)

    successes = sum(1 for r in results if r["success"])
    avg_length = sum(r["turns"] for r in results) / len(results)
    success_rate = successes / len(results) * 100

    print(f"\n--- {policy_name} ---")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Dialogue Length: {avg_length:.1f} turns")
    print(f"Timeout Rate: {100 - success_rate:.1f}%")

    return results


# Load models and evaluate
all_results = {}

for name, model_path in [
    ("Sparse", "uber_policy_sparse"),
    ("Version 2 (LLM whole)", "uber_policy_llm_v2"),
    ("Version 3 (Hybrid)", "uber_policy_llm_v3"),
    ("Version 4 (Distilled Hybrid)", "uber_policy_llm_v4")
]:
    env = UberDialogueEnv()
    model = PPO.load(model_path, env=env)
    results = evaluate_policy(model, name)
    all_results[name] = results


with open("test_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nAll dialogues saved to test_results.json")