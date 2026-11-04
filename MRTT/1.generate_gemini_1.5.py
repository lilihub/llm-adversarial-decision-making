import numpy as np
import pandas as pd
import random
import math
import re
import google.generativeai as genai


genai.configure(api_key= "YOUR_API_KEY")
def act(prompt):
    model=genai.GenerativeModel(
                            model_name="gemini-1.5-flash",
                            system_instruction="You are an investor in a Trust Game with 10 rounds. You will receive 20 monetary units in each round and then decide how much to invest (0-20 units). The invested amount will be tripled and given to a trustee, who then decides how much to return to you. You will be provided with how much the trustee kept and repaied you back, followed by your total earnings (the money you kept plus the repayment of the trustee) for the entire round. Your goal is to maximize your earnings. You will be asked how much you want to invest in each round. Answer with only one single number that you want to invest.")
    response = model.generate_content(prompt)
    return response.candidates[0].content.parts[0].text.strip().upper()

num_runs=200
rounds = 10
initial_units = 20
for run in range(192, num_runs):
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
    df.to_csv('sim_data/gemini/gemini_1.5/experiment_' + str(run) + '.csv')
