import random

# All possible values for each slot
SLOT_VALUES = {
    "pickup":   ["USC", "Downtown", "Santa Monica"],
    "dropoff":  ["LAX", "Hollywood", "Venice Beach"],
    "ridetype": ["UberX", "UberPool", "Uber Black"],
    "time":     ["now", "in 30 minutes", "in 1 hour"]
}

class SimulatedUser:
    def __init__(self):
        self.goal = None

    def reset(self):
        # Assign a random hidden goal at the start of each episode
        # The system doesn't know this — it has to discover it through dialogue
        self.goal = {
            "pickup":   random.choice(SLOT_VALUES["pickup"]),
            "dropoff":  random.choice(SLOT_VALUES["dropoff"]),
            "ridetype": random.choice(SLOT_VALUES["ridetype"]),
            "time":     random.choice(SLOT_VALUES["time"])
        }

    def respond(self, system_action, current_values):
        # current_values is a dict of what the system currently has filled
        # e.g. {"pickup": "USC", "dropoff": None, "ridetype": None, "time": None}
        # The user checks this when deciding to confirm yes or no

        # --- REQUEST actions ---
        # System is asking the user to provide a slot value
        # User either provides it (80%) or says something irrelevant (20%) this is to simulate noise in the dialogue and make it more realistic

        if system_action == "request_pickup":
            return random.choices(
                ["provide_pickup", "irrelevant"],
                weights=[80, 20]
            )[0] 

        elif system_action == "request_dropoff":
            return random.choices(
                ["provide_dropoff", "irrelevant"],
                weights=[80, 20]
            )[0]

        elif system_action == "request_ridetype":
            return random.choices(
                ["provide_ridetype", "irrelevant"],
                weights=[80, 20]
            )[0]

        elif system_action == "request_time":
            return random.choices(
                ["provide_time", "irrelevant"],
                weights=[80, 20]
            )[0]

        # --- CONFIRM actions ---
        # System is asking the user to confirm a value it already filled
        # User says yes ONLY if the system's value matches the hidden goal
        # Otherwise user says no (the system got it wrong)

        elif system_action == "request_confirm_pickup":
            if current_values["pickup"] == self.goal["pickup"]:
                return random.choices(
                    ["confirm_pos_pickup", "irrelevant"],
                    weights=[80, 20]
                )[0]
            else:
                return random.choices(
                    ["confirm_neg_pickup", "irrelevant"],
                    weights=[80, 20]
                )[0]

        elif system_action == "request_confirm_dropoff":
            if current_values["dropoff"] == self.goal["dropoff"]:
                return random.choices(
                    ["confirm_pos_dropoff", "irrelevant"],
                    weights=[80, 20]
                )[0]
            else:
                return random.choices(
                    ["confirm_neg_dropoff", "irrelevant"],
                    weights=[80, 20]
                )[0]

        elif system_action == "request_confirm_ridetype":
            if current_values["ridetype"] == self.goal["ridetype"]:
                return random.choices(
                    ["confirm_pos_ridetype", "irrelevant"],
                    weights=[80, 20]
                )[0]
            else:
                return random.choices(
                    ["confirm_neg_ridetype", "irrelevant"],
                    weights=[80, 20]
                )[0]

        elif system_action == "request_confirm_time":
            if current_values["time"] == self.goal["time"]:
                return random.choices(
                    ["confirm_pos_time", "irrelevant"],
                    weights=[80, 20]
                )[0]
            else:
                return random.choices(
                    ["confirm_neg_time", "irrelevant"],
                    weights=[80, 20]
                )[0]

        else:
            return "irrelevant"


# ---- Test it ----
if __name__ == "__main__":
    user = SimulatedUser()
    user.reset()

    print("Hidden goal:", user.goal)

    # Simulate what happens when the system fills pickup correctly
    current_values = {
        "pickup":   user.goal["pickup"],  # system got it right
        "dropoff":  None,
        "ridetype": None,
        "time":     None
    }
    print("\nSystem asks to confirm pickup (correct value):")
    for _ in range(5):
        print(" ", user.respond("request_confirm_pickup", current_values))

    # Simulate what happens when the system fills pickup incorrectly
    wrong_values = {
        "pickup":   "Hollywood",  # wrong value
        "dropoff":  None,
        "ridetype": None,
        "time":     None
    }
    print("\nSystem asks to confirm pickup (WRONG value):")
    for _ in range(5):
        print(" ", user.respond("request_confirm_pickup", wrong_values))