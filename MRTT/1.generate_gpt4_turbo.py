import numpy as np
import pandas as pd
import random
import math
import re
from openai import OpenAI


client = OpenAI(api_key= "your-openai-api-key")
def act(text):
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an investor in a Trust Game with 10 rounds. You will receive 20 monetary units in each round and then decide how much to invest (0-20 units). The invested amount will be tripled and given to a trustee, who then decides how much to return to you. You will be provided with how much the trustee kept and repaied you back, followed by your total earnings (the money you kept plus the repayment of the trustee) for the entire round. Your goal is to maximize your earnings. You will be asked how much you want to invest in each round. Answer with only one single number that you want to invest."},
            {"role": "user", "content": text}
            ]
    )
    response = completion.choices[0].message.content.strip().upper()
    return response

num_runs=10
rounds = 10
initial_units = 20
for run in range(0, num_runs):
    previous_interactions= []
    data = []
    for round_number in range(1, rounds + 1):
        # Construct the prompt for ChatGPT
        total_text = ""
        if len(previous_interactions) > 0:
            total_text = "Your previous investment records are as follows:\n"
            for count, interaction in enumerate(previous_interactions):
                total_text += interaction

        total_text += "Q: You have 20 monetary units. How much do you want to invest in this round?\n "
        while True:
            response = act(total_text)
            print(response)
            match = re.search(r'\d+', response)
            if match:
                if 0 <= int(match.group()) <=20:
                    investment = str(match.group())
                    total_text += " " + investment + ".\n"
                    print(total_text)
                    break
        
        # Triple the investment for the trustee
        tripled_investment = int(investment) * 3
        
        # Trustee's random repayment (example logic)
        repayment = math.floor(random.random() * tripled_investment)
        kept = tripled_investment - repayment
        
        # Calculate round earnings and update total
        round_earnings = initial_units - int(investment) + repayment

        feedback_item = "- In Round " + str(round_number) + ", you invested " + investment + ". The trustee kept " + str(kept) + " and repaid you " + str(repayment) + ". Your total earning of this round is " + str(round_earnings) + "\n"
        
        previous_interactions.append(feedback_item)


        # row = [run, trial, index_action, treasure, reward_probs[0], reward_probs[1]]
        row = [run, round_number, int(investment), repayment, round_earnings]
        data.append(row)       

    df = pd.DataFrame(data, columns=['run', 'round', 'investment', 'repay', 'round_earnings'])
    df.to_csv('sim_data/gpt/gpt-4-turbo/experiment_' + str(run) + '.csv')
