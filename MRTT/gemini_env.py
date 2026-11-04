import numpy as np
from util.helper import one_hot
from openai import OpenAI
import re
import google.generativeai as genai

genai.configure(api_key= "YOUR_API_KEY")

class Gemini_ENV:
    def __init__(self, Gemini_version = 'gemini-1.5-flash'):
        self.GPT = Gemini_version
        self.reset()

    def reset(self):
        self.previous_interactions = []
        self.round_number = 0
 

    def act(self, text):
        model=genai.GenerativeModel(
                            model_name=self.GPT,
                            system_instruction="You are an investor in a Trust Game with 10 rounds. You will receive 20 monetary units in each round and then decide how much to invest (0-20 units). The invested amount will be tripled and given to a trustee, who then decides how much to return to you. You will be provided with how much the trustee kept and repaied you back, followed by your total earnings (the money you kept plus the repayment of the trustee) for the entire round. Your goal is to maximize your earnings. You will be asked how much you want to invest in each round. Answer with only one single number that you want to invest.")
        response = model.generate_content(text)
        return response.candidates[0].content.parts[0].text.strip().upper()
    
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
