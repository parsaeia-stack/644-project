import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from simulated_user import SimulatedUser, SLOT_VALUES

# Map action numbers to action names
# SB3 works with numbers (0-7), not strings
ACTIONS = [
    "request_pickup",           # 0
    "request_dropoff",          # 1
    "request_ridetype",         # 2
    "request_time",             # 3
    "request_confirm_pickup",   # 4
    "request_confirm_dropoff",  # 5
    "request_confirm_ridetype", # 6
    "request_confirm_time"      # 7
]

class UberDialogueEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.user = SimulatedUser()

        # What SB3 sees: 8 binary values (FILLED and CONF for each slot)
        self.observation_space = spaces.MultiBinary(8)

        # What SB3 can do: pick one of 8 actions
        self.action_space = spaces.Discrete(8)

        self.max_turns = 30  # episode times out after 30 turns

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the simulated user with a new random goal
        self.user.reset()

        # Reset the state — nothing filled or confirmed yet
        self.state = {
            "pickup_filled": 0, "pickup_conf": 0,
            "dropoff_filled": 0, "dropoff_conf": 0,
            "ridetype_filled": 0, "ridetype_conf": 0,
            "time_filled": 0, "time_conf": 0
        }

        # Reset the current values the system has heard
        self.current_values = {
            "pickup": None, "dropoff": None,
            "ridetype": None, "time": None
        }

        self.turn = 0
        self.history = []  # will be used later for LLM reward

        return self._get_obs(), {}

    def step(self, action):
        action_name = ACTIONS[action]
        self.turn += 1

        # Get simulated user response
        user_response = self.user.respond(action_name, self.current_values)

        # Update state based on user response 
        state_before = self._get_obs().tolist()  # capture BEFORE update
        self._update_state(action_name, user_response)
        
        # Log turn for LLM reward later
        self.history.append({
            "turn": self.turn,
            "action": action_name,
            "user_response": user_response,
            "state": state_before
        })

        # Check if task is complete
        done = self._is_done()
        timeout = self.turn >= self.max_turns

        # Sparse reward
        reward = -5
        if done:
            reward += 500

        return self._get_obs(), reward, done or timeout, False, {}

    def _update_state(self, action_name, user_response):
        # Handle provide responses — slot gets filled with a random value from goal
        if user_response == "provide_pickup":
            self.current_values["pickup"] = self.user.goal["pickup"]
            self.state["pickup_filled"] = 1
            self.state["pickup_conf"] = 0  # reset conf when refilled

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

        # Handle positive confirmations
        elif user_response == "confirm_pos_pickup":
            self.state["pickup_conf"] = 1

        elif user_response == "confirm_pos_dropoff":
            self.state["dropoff_conf"] = 1

        elif user_response == "confirm_pos_ridetype":
            self.state["ridetype_conf"] = 1

        elif user_response == "confirm_pos_time":
            self.state["time_conf"] = 1

        # Handle negative confirmations — slot goes back to empty
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

        # irrelevant — nothing changes

    def _is_done(self):
        # Task is complete when all 4 slots are both filled and confirmed
        return (
            self.state["pickup_filled"] == 1 and self.state["pickup_conf"] == 1 and
            self.state["dropoff_filled"] == 1 and self.state["dropoff_conf"] == 1 and
            self.state["ridetype_filled"] == 1 and self.state["ridetype_conf"] == 1 and
            self.state["time_filled"] == 1 and self.state["time_conf"] == 1
        )

    def _get_obs(self):
        # Return state as numpy array of 8 binary values — this is what SB3 sees
        return np.array([
            self.state["pickup_filled"],   self.state["pickup_conf"],
            self.state["dropoff_filled"],  self.state["dropoff_conf"],
            self.state["ridetype_filled"], self.state["ridetype_conf"],
            self.state["time_filled"],     self.state["time_conf"]
        ], dtype=np.int8)


# ---- Test it ----
if __name__ == "__main__":
    env = UberDialogueEnv()
    obs, _ = env.reset()

    print("User goal:", env.user.goal)
    print("Initial state:", obs)
    print()

    # Manually take a few actions and see what happens
    test_actions = [0, 4, 1, 5, 2, 6, 3, 7]  # request then confirm each slot
    for action in test_actions:
        obs, reward, done, _, _ = env.step(action)
        print(f"Action: {ACTIONS[action]}")
        print(f"User response: {env.history[-1]['user_response']}")
        print(f"State: {obs}  Reward: {reward}  Done: {done}")
        print()
        if done:
            print("Task completed!")
            break