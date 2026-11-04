import numpy as np
import pandas as pd
import google.generativeai as genai


genai.configure(api_key= "YOUR_API_KEY")
def act(prompt):
    model=genai.GenerativeModel(
                            model_name="gemini-1.5-pro",
                            system_instruction="You are a space explorer in a game. Your task is to choose between visiting Planet X or Planet Y in each round, aiming to find as many gold coins as possible. The probability of finding gold coins on each planet is unknown at the start, but you can learn and adjust your strategy based on the outcomes of your previous visits. Respond with one single word 'X' for Planet X or 'Y' for Planet Y.\n")
    response = model.generate_content(prompt)
    return response.candidates[0].content.parts[0].text.strip().upper()

action_to_index = {"X": 0, "Y": 1}
num_runs = 2
num_trials = 100

# reward_probs = np.random.uniform(0.25, 0.25, (2,))
reward_probs = 0.25
for run in range(0, num_runs):
    previous_interactions = []
    data = []  
    for trial in range(num_trials):
        total_text = ""
        if len(previous_interactions) > 0:
            total_text = "Your previous space travels went as follows:\n"
            # total_text = feedback_item
        for count, interaction in enumerate(previous_interactions):
            # total_text += "- " + str(len(previous_interactions) - count) + days + " ago, "
            total_text += interaction

        total_text += "Q: Which planet do you want to go to in Trial " + str(trial + 1) + "?\nA: Planet"

        while True:
            action = act(total_text)
            if action in action_to_index:
                index_action = action_to_index[action]
                total_text += " " + action + ".\n"
                print(total_text)
                break
        
        # treasure = np.random.binomial(1, reward_probs[index_action], 1)[0]
        treasure = np.random.binomial(1, reward_probs, 1)[0]
        feedback_item = "- In Trial " + str(trial + 1) + ", you went to planet " + action + " and found " + ("100 gold coins." if treasure else "nothing.") + "\n"
        previous_interactions.append(feedback_item)

        # row = [run, trial, index_action, treasure, reward_probs[0], reward_probs[1]]
        row = [run, trial, index_action, treasure, reward_probs, reward_probs]
        data.append(row)       

    df = pd.DataFrame(data, columns=['run', 'trial', 'action', 'reward', 'probsX', 'probsY'])
    df.to_csv('bandit/sim_data/gemini/gemini_1.5_pro/experiment_' + str(run) + '.csv')
