import numpy as np
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key= "your-openai-api-key")
def act(text):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a space explorer in a game. Your task is to choose between visiting Planet X or Planet Y in each round, aiming to find as many gold coins as possible. The probability of finding gold coins on each planet is unknown at the start, but you can learn and adjust your strategy based on the outcomes of your previous visits. Respond with one single word 'X' for Planet X or 'Y' for Planet Y."},
            {"role": "user", "content": text}
            ],
         max_tokens=1
    )

    response = completion.choices[0].message.content.strip().upper()
    return response

action_to_index = {"X": 0, "Y": 1}
num_runs = 200
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

        total_text += "Q: Which planet do you want to go to in Trial " + str(trial + 1) + "?\nA: Planet "

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
    df.to_csv('sim_data/gpt/gpt-3.5-turbo/experiment_' + str(run) + '.csv')
