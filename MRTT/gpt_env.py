import numpy as np
from util.helper import one_hot
from openai import OpenAI
import re
client = OpenAI(api_key= "YOUR_API_KEY")


class GPT_ENV:
    def __init__(self, GPT_version = 'gpt_3.5_turbo_1106'):
        self.GPT = GPT_version
        self.reset()

    def reset(self):
        self.previous_interactions = []
        self.round_number = 0
 
    def act(self, text):
        completion = client.chat.completions.create(
        model=self.GPT,
        messages=[
            {"role": "system", "content": "You are an investor in a Trust Game with 10 rounds. You will receive 20 monetary units in each round and then decide how much to invest (0-20 units). The invested amount will be tripled and given to a trustee, who then decides how much to return to you. The results of this will be provided to you, followed by the total earnings (the money you kept plus the repayment of the trustee) for the entire round. Your goal is to maximize your earnings. You will be asked how much you want to invest in each round. Answer with only one single number that you want to invest."},
            {"role": "user", "content": text}
            ],
         max_tokens=8
        )
        response = completion.choices[0].message.content.strip().upper()
        return response     

    def step(self, investment, kept, repayment, round_earnings):
        # treasure = np.random.binomial(1, self.reward_probs, 1)[0]
        if self.round_number >0:
            feedback_item = "- In Round " + str(self.round_number) + ", you invested " + str(investment.astype(int).item()) + ". The trustee kept " + str(kept.astype(int).item()) + " and repaid you " + str(repayment.astype(int).item()) + ". Your total earning of this round is " + str(round_earnings.astype(int).item()) + "\n"
            self.previous_interactions.append(feedback_item)


        # Construct the prompt for ChatGPT
        total_text = ""
        if len(self.previous_interactions) > 0:
            total_text = "Your previous investment records are as follows:\n"
            for count, interaction in enumerate(self.previous_interactions):
                total_text += interaction

        total_text += "Q: You have 20 monetary units. How much do you want to invest in this round?\n "
        while True:
            response = self.act(total_text)
            match = re.search(r'\d+', response)
            if match:
                if 0 <= int(match.group()) <=20:
                    investment = str(match.group())
                    total_text += " " + investment + ".\n"
                    print(total_text)
                    break


        self.round_number += 1
        return np.array([int(investment)],np.float32)

    


if __name__ == '__main__':
    gpt = GPT_ENV()
    a = gpt.reset()
