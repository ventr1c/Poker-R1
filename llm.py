from vllm import LLM, SamplingParams

class LLMPlayer:
    def __init__(self, model_path="/mnt/data/hf_models/DeepSeek-R1-Distill-Qwen-7B", temperature=0.8, top_p=0.95):
        """
        Initializes the LLM-based poker player.
        
        Args:
        - model_path (str): Path to the preloaded model.
        - temperature (float): Sampling temperature for LLM.
        - top_p (float): Top-p sampling for nucleus sampling.
        """
        self.llm = LLM(model=model_path)
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p)

    def format_prompt(self, game_state):
        """
        Formats the poker game state into a natural language prompt.

        Args:
        - game_state (dict): Dictionary containing relevant poker state information.

        Returns:
        - prompt (str): Formatted text prompt for LLM.
        """
        prompt = (
            f"Poker game state:\n"
            f"- Hole cards: {game_state['hole_cards']}\n"
            f"- Community cards: {game_state['community_cards']}\n"
            f"- Pot size: {game_state['pot']}\n"
            f"- Stacks: {game_state['stacks']}\n"
            f"- Call amount: {game_state['call']}\n"
            f"- Min raise: {game_state['min_raise']}\n"
            f"- Max raise: {game_state['max_raise']}\n\n"
            f"Decide the best action: 'fold', 'call', or 'raise X', where X is the raise amount."
        )
        return prompt

    def decide_action(self, game_state):
        """
        Uses the LLM to generate a poker action based on the current game state.

        Args:
        - game_state (dict): Dictionary containing relevant poker state information.

        Returns:
        - action (str): Valid poker action ('fold', 'call', or 'raise X').
        """
        prompt = self.format_prompt(game_state)
        outputs = self.llm.generate([prompt], self.sampling_params)
        raw_output = outputs[0].outputs[0].text.strip().lower()

        # TODO Convert the action to the int bet.
        # Extract action
        if "fold" in raw_output:
            return "fold"
        elif "call" in raw_output:
            return "call"
        elif "raise" in raw_output:
            try:
                raise_amount = int(raw_output.split("raise")[1].strip())
                # Ensure the raise amount is within limits
                if game_state["min_raise"] <= raise_amount <= game_state["max_raise"]:
                    return f"raise {raise_amount}"
            except ValueError:
                pass  # If parsing fails, fallback to a default action

        # Fallback action if parsing fails
        return "call"

if __name__ == '__main__':
    prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    
    llm = LLM(model="/mnt/data/hf_models/DeepSeek-R1-Distill-Qwen-7B")
    
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")