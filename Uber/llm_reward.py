from google import genai
import json
import os

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def get_llm_rewards(history):
    transcript = ""
    for h in history:
        transcript += f"Turn {h['turn']}:\n"
        transcript += f"  State: {h['state']}\n"
        transcript += f"  System action: {h['action']}\n"
        transcript += f"  User response: {h['user_response']}\n\n"

    prompt = f"""You are evaluating a task-oriented dialogue system booking an Uber ride.
The system must fill and confirm 6 slots: PICKUP, DROPOFF, RIDETYPE, TIME, PASSENGERS, PAYMENT.

State format: [pickup_filled, pickup_conf, dropoff_filled, dropoff_conf, ridetype_filled, ridetype_conf, time_filled, time_conf, passengers_filled, passengers_conf, payment_filled, payment_conf]
1 = yes, 0 = no

Scoring rules:
- +1.0: action fills or confirms a slot that needed it (correct and necessary)
- +0.5: reasonable action but not the most urgent given the current state
- -0.5: action is premature (e.g. confirming a slot before filling it)
- -1.0: action is clearly wrong (e.g. requesting a slot already filled and confirmed, or confirming unfilled slot)
- 0.0: user said irrelevant, nothing changed, action was neutral

Here is the full dialogue:

{transcript}

Return ONLY a JSON list of scores, one per turn, in order.
Do not include any explanation."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    response_text = response.text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]

    scores = json.loads(response_text)
    return scores


if __name__ == "__main__":
    test_history = [
        {"turn": 1, "action": "request_pickup", "user_response": "provide_pickup", "state": [0,0,0,0,0,0,0,0,0,0,0,0]},
        {"turn": 2, "action": "request_confirm_pickup", "user_response": "confirm_pos_pickup", "state": [1,0,0,0,0,0,0,0,0,0,0,0]},
        {"turn": 3, "action": "request_confirm_dropoff", "user_response": "confirm_neg_dropoff", "state": [1,1,0,0,0,0,0,0,0,0,0,0]},
    ]

    scores = get_llm_rewards(test_history)
    print("LLM scores:", scores)
    for i, (h, s) in enumerate(zip(test_history, scores)):
        print(f"Turn {i+1} — {h['action']}: {s}")