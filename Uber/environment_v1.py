from environment import UberDialogueEnv, ACTIONS
from llm_reward import get_llm_rewards

class UberDialogueEnvV1(UberDialogueEnv):
    """
    Version 1: LLM turn-by-turn reward.
    After each turn, send the conversation so far to the LLM
    and use its score as the reward for that turn.
    """
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
        """
         # Sparse reward
        reward = -5
        if done:
            reward += 500
        
        """
        # LLM reward
        # Get LLM reward for the conversation so far
        try:
            scores = get_llm_rewards(self.history)
            reward = scores[-1]  # score for the most recent turn
        except Exception as e:
            print(f"LLM call failed: {e}, falling back to sparse")
            reward = -5
            if done:
                reward += 500
        
        
        print(f"Turn {self.turn} | Action: {action_name} | Reward: {reward:.2f}")
       

        return self._get_obs(), reward, done or timeout, False, {}

# Test it
if __name__ == "__main__":
    env = UberDialogueEnvV1()
    obs, _ = env.reset()
    print("User goal:", env.user.goal)

    test_actions = [0, 4, 1, 5, 2, 6, 3, 7]
    for action in test_actions:
        obs, reward, done, _, _ = env.step(action)
        last = env.history[-1]
        print(f"Action: {last['action']} | User: {last['user_response']} | Reward: {reward:.2f} | Done: {done}")
        if done:
            print("Task completed!")
            break