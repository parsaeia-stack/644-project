import random

SLOT_VALUES = {
    "pickup":     ["USC", "Downtown", "Santa Monica", "Hollywood", "Koreatown", "Westwood", "Silver Lake", "Echo Park"],
    "dropoff":    ["LAX", "Hollywood", "Venice Beach", "Burbank Airport", "Long Beach", "Pasadena", "Malibu", "Santa Barbara"],
    "ridetype":   ["UberX", "UberPool", "Uber Black", "UberXL", "Uber Comfort"],
    "time":       ["now", "in 30 minutes", "in 1 hour", "in 2 hours", "tomorrow morning", "tomorrow evening"],
    "passengers": ["1", "2", "3", "4"],
    "payment":    ["cash", "card", "PayPal"]
}

class SimulatedUser:
    def __init__(self):
        self.goal = None

    def reset(self):
        self.goal = {
            "pickup":     random.choice(SLOT_VALUES["pickup"]),
            "dropoff":    random.choice(SLOT_VALUES["dropoff"]),
            "ridetype":   random.choice(SLOT_VALUES["ridetype"]),
            "time":       random.choice(SLOT_VALUES["time"]),
            "passengers": random.choice(SLOT_VALUES["passengers"]),
            "payment":    random.choice(SLOT_VALUES["payment"])
        }

    def _wrong_value(self, slot):
        # Return a random value that is NOT the correct goal value
        options = [v for v in SLOT_VALUES[slot] if v != self.goal[slot]]
        return random.choice(options)

    def respond(self, system_action, current_values):

        # --- REQUEST actions ---
        if system_action == "request_pickup":
            return random.choices(
                ["provide_pickup", "provide_wrong_pickup", "irrelevant"],
                weights=[70, 10, 20]
            )[0]

        elif system_action == "request_dropoff":
            return random.choices(
                ["provide_dropoff", "provide_wrong_dropoff", "irrelevant"],
                weights=[70, 10, 20]
            )[0]

        elif system_action == "request_ridetype":
            return random.choices(
                ["provide_ridetype", "provide_wrong_ridetype", "irrelevant"],
                weights=[70, 10, 20]
            )[0]

        elif system_action == "request_time":
            return random.choices(
                ["provide_time", "provide_wrong_time", "irrelevant"],
                weights=[70, 10, 20]
            )[0]

        elif system_action == "request_passengers":
            return random.choices(
                ["provide_passengers", "provide_wrong_passengers", "irrelevant"],
                weights=[70, 10, 20]
            )[0]

        elif system_action == "request_payment":
            return random.choices(
                ["provide_payment", "provide_wrong_payment", "irrelevant"],
                weights=[70, 10, 20]
            )[0]

        # --- CONFIRM actions ---
        elif system_action == "request_confirm_pickup":
            if current_values["pickup"] == self.goal["pickup"]:
                return random.choices(["confirm_pos_pickup", "irrelevant"], weights=[80, 20])[0]
            else:
                return random.choices(["confirm_neg_pickup", "irrelevant"], weights=[80, 20])[0]

        elif system_action == "request_confirm_dropoff":
            if current_values["dropoff"] == self.goal["dropoff"]:
                return random.choices(["confirm_pos_dropoff", "irrelevant"], weights=[80, 20])[0]
            else:
                return random.choices(["confirm_neg_dropoff", "irrelevant"], weights=[80, 20])[0]

        elif system_action == "request_confirm_ridetype":
            if current_values["ridetype"] == self.goal["ridetype"]:
                return random.choices(["confirm_pos_ridetype", "irrelevant"], weights=[80, 20])[0]
            else:
                return random.choices(["confirm_neg_ridetype", "irrelevant"], weights=[80, 20])[0]

        elif system_action == "request_confirm_time":
            if current_values["time"] == self.goal["time"]:
                return random.choices(["confirm_pos_time", "irrelevant"], weights=[80, 20])[0]
            else:
                return random.choices(["confirm_neg_time", "irrelevant"], weights=[80, 20])[0]

        elif system_action == "request_confirm_passengers":
            if current_values["passengers"] == self.goal["passengers"]:
                return random.choices(["confirm_pos_passengers", "irrelevant"], weights=[80, 20])[0]
            else:
                return random.choices(["confirm_neg_passengers", "irrelevant"], weights=[80, 20])[0]

        elif system_action == "request_confirm_payment":
            if current_values["payment"] == self.goal["payment"]:
                return random.choices(["confirm_pos_payment", "irrelevant"], weights=[80, 20])[0]
            else:
                return random.choices(["confirm_neg_payment", "irrelevant"], weights=[80, 20])[0]

        else:
            return "irrelevant"


if __name__ == "__main__":
    user = SimulatedUser()
    user.reset()
    print("Hidden goal:", user.goal)

    current_values = {
        "pickup": None, "dropoff": None,
        "ridetype": None, "time": None,
        "passengers": None, "payment": None
    }
    print("\nRequest pickup (may get wrong value):")
    for _ in range(5):
        print(" ", user.respond("request_pickup", current_values))