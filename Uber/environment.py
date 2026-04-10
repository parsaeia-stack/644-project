import gymnasium as gym
import numpy as np
from gymnasium import spaces
from simulated_user import SimulatedUser, SLOT_VALUES

ACTIONS = [
    "request_pickup",             # 0
    "request_dropoff",            # 1
    "request_ridetype",           # 2
    "request_time",               # 3
    "request_passengers",         # 4
    "request_payment",            # 5
    "request_confirm_pickup",     # 6
    "request_confirm_dropoff",    # 7
    "request_confirm_ridetype",   # 8
    "request_confirm_time",       # 9
    "request_confirm_passengers", # 10
    "request_confirm_payment"     # 11
]

class UberDialogueEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.user = SimulatedUser()
        self.action_names = ACTIONS

        # 6 slots × 2 flags = 12 binary values
        self.observation_space = spaces.MultiBinary(12)

        # 12 possible actions
        self.action_space = spaces.Discrete(12)

        self.max_turns = 50

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.user.reset()

        self.state = {
            "pickup_filled": 0,     "pickup_conf": 0,
            "dropoff_filled": 0,    "dropoff_conf": 0,
            "ridetype_filled": 0,   "ridetype_conf": 0,
            "time_filled": 0,       "time_conf": 0,
            "passengers_filled": 0, "passengers_conf": 0,
            "payment_filled": 0,    "payment_conf": 0,
        }

        self.current_values = {
            "pickup": None, "dropoff": None,
            "ridetype": None, "time": None,
            "passengers": None, "payment": None
        }

        self.turn = 0
        self.history = []

        return self._get_obs(), {}

    def step(self, action):
        action_name = self.action_names[action]
        self.turn += 1

        user_response = self.user.respond(action_name, self.current_values)

        state_before = self._get_obs().tolist()
        self._update_state(action_name, user_response)

        self.history.append({
            "turn": self.turn,
            "action": action_name,
            "user_response": user_response,
            "state": state_before
        })

        done = self._is_done()
        timeout = self.turn >= self.max_turns

        reward = -5
        if done:
            reward += 500

        return self._get_obs(), reward, done or timeout, False, {}

    def _update_state(self, action_name, user_response):
        # CORRECT VALUE responses
        if user_response == "provide_pickup":
            self.current_values["pickup"] = self.user.goal["pickup"]
            self.state["pickup_filled"] = 1
            self.state["pickup_conf"] = 0

        elif user_response == "provide_dropoff":
            self.current_values["dropoff"] = self.user.goal["dropoff"]
            self.state["dropoff_filled"] = 1
            self.state["dropoff_conf"] = 0

        elif user_response == "provide_ridetype":
            self.current_values["ridetype"] = self.user.goal["ridetype"]
            self.state["ridetype_filled"] = 1
            self.state["ridetype_conf"] = 0

        elif user_response == "provide_time":
            self.current_values["time"] = self.user.goal["time"]
            self.state["time_filled"] = 1
            self.state["time_conf"] = 0

        elif user_response == "provide_passengers":
            self.current_values["passengers"] = self.user.goal["passengers"]
            self.state["passengers_filled"] = 1
            self.state["passengers_conf"] = 0

        elif user_response == "provide_payment":
            self.current_values["payment"] = self.user.goal["payment"]
            self.state["payment_filled"] = 1
            self.state["payment_conf"] = 0

        # WRONG VALUE responses
        elif user_response == "provide_wrong_pickup":
            self.current_values["pickup"] = self.user._wrong_value("pickup")
            self.state["pickup_filled"] = 1
            self.state["pickup_conf"] = 0

        elif user_response == "provide_wrong_dropoff":
            self.current_values["dropoff"] = self.user._wrong_value("dropoff")
            self.state["dropoff_filled"] = 1
            self.state["dropoff_conf"] = 0

        elif user_response == "provide_wrong_ridetype":
            self.current_values["ridetype"] = self.user._wrong_value("ridetype")
            self.state["ridetype_filled"] = 1
            self.state["ridetype_conf"] = 0

        elif user_response == "provide_wrong_time":
            self.current_values["time"] = self.user._wrong_value("time")
            self.state["time_filled"] = 1
            self.state["time_conf"] = 0

        elif user_response == "provide_wrong_passengers":
            self.current_values["passengers"] = self.user._wrong_value("passengers")
            self.state["passengers_filled"] = 1
            self.state["passengers_conf"] = 0

        elif user_response == "provide_wrong_payment":
            self.current_values["payment"] = self.user._wrong_value("payment")
            self.state["payment_filled"] = 1
            self.state["payment_conf"] = 0

        # POSITIVE CONFIRM responses
        elif user_response == "confirm_pos_pickup":
            self.state["pickup_conf"] = 1

        elif user_response == "confirm_pos_dropoff":
            self.state["dropoff_conf"] = 1

        elif user_response == "confirm_pos_ridetype":
            self.state["ridetype_conf"] = 1

        elif user_response == "confirm_pos_time":
            self.state["time_conf"] = 1

        elif user_response == "confirm_pos_passengers":
            self.state["passengers_conf"] = 1

        elif user_response == "confirm_pos_payment":
            self.state["payment_conf"] = 1

        # NEGATIVE CONFIRM responses — reset slot
        elif user_response == "confirm_neg_pickup":
            self.state["pickup_filled"] = 0
            self.state["pickup_conf"] = 0
            self.current_values["pickup"] = None

        elif user_response == "confirm_neg_dropoff":
            self.state["dropoff_filled"] = 0
            self.state["dropoff_conf"] = 0
            self.current_values["dropoff"] = None

        elif user_response == "confirm_neg_ridetype":
            self.state["ridetype_filled"] = 0
            self.state["ridetype_conf"] = 0
            self.current_values["ridetype"] = None

        elif user_response == "confirm_neg_time":
            self.state["time_filled"] = 0
            self.state["time_conf"] = 0
            self.current_values["time"] = None

        elif user_response == "confirm_neg_passengers":
            self.state["passengers_filled"] = 0
            self.state["passengers_conf"] = 0
            self.current_values["passengers"] = None

        elif user_response == "confirm_neg_payment":
            self.state["payment_filled"] = 0
            self.state["payment_conf"] = 0
            self.current_values["payment"] = None

        # irrelevant — nothing changes

    def _is_done(self):
        return (
            self.state["pickup_filled"] == 1    and self.state["pickup_conf"] == 1 and
            self.state["dropoff_filled"] == 1   and self.state["dropoff_conf"] == 1 and
            self.state["ridetype_filled"] == 1  and self.state["ridetype_conf"] == 1 and
            self.state["time_filled"] == 1      and self.state["time_conf"] == 1 and
            self.state["passengers_filled"] == 1 and self.state["passengers_conf"] == 1 and
            self.state["payment_filled"] == 1   and self.state["payment_conf"] == 1
        )

    def _get_obs(self):
        return np.array([
            self.state["pickup_filled"],     self.state["pickup_conf"],
            self.state["dropoff_filled"],    self.state["dropoff_conf"],
            self.state["ridetype_filled"],   self.state["ridetype_conf"],
            self.state["time_filled"],       self.state["time_conf"],
            self.state["passengers_filled"], self.state["passengers_conf"],
            self.state["payment_filled"],    self.state["payment_conf"],
        ], dtype=np.int8)


if __name__ == "__main__":
    env = UberDialogueEnv()
    obs, _ = env.reset()
    print("User goal:", env.user.goal)
    print("Initial state:", obs)

    test_actions = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
    for action in test_actions:
        obs, reward, done, _, _ = env.step(action)
        print(f"Action: {ACTIONS[action]}")
        print(f"User: {env.history[-1]['user_response']}")
        print(f"State: {obs} | Reward: {reward} | Done: {done}")
        print()
        if done:
            print("Task completed!")
            break