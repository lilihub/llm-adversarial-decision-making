import numpy as np
from util.helper import one_hot
from openai import OpenAI
client = OpenAI(api_key= "your-openai-api-key")
# engine = "gpt-3.5-turbo-instruct"


class GPT_ENV:
    def __init__(self, GPT_version = 'gpt_3.5_turbo_1106'):
        self.action_to_index = {"X": 0, "Y": 1}
        self.reward_probs = 0.25
        self.GPT = GPT_version
        self.reset()

    def reset(self):
        self.previous_interactions = []
        self.trial = 1
 
    def act(self, text):
        completion = client.chat.completions.create(
        model=self.GPT,
        messages=[
            {"role": "system", "content": "You are a space explorer in a game. Your task is to choose between visiting Planet X or Planet Y in each round, aiming to find as many gold coins as possible. The probability of finding gold coins on each planet is unknown at the start, but you can learn and adjust your strategy based on the outcomes of your previous visits. Respond with one single word 'X' for Planet X or 'Y' for Planet Y."},
            {"role": "user", "content": text}
            ],
         max_tokens=1
        )
        response = completion.choices[0].message.content.strip().upper()
        return response        

    def step(self, action_old, treasure):
        # treasure = np.random.binomial(1, self.reward_probs, 1)[0]
        if self.trial >1:
            if action_old[0][0] == 1:
                action_old = "X"
            elif action_old[0][1] == 1:
                action_old = "Y"
            else:
                action_old = ""
            feedback_item = "- In Trial " + str(self.trial-1) + ", you went to planet " + action_old + " and found " + ("100 gold coins." if treasure else "nothing.") + "\n"
            self.previous_interactions.append(feedback_item)

        total_text = ""
        if len(self.previous_interactions) > 0:
            total_text = "Your previous space travels went as follows:\n"
        for count, interaction in enumerate(self.previous_interactions):
            total_text += interaction

        total_text += "Q: Which planet do you want to go to in Trial " + str(self.trial) + "?\nA: Planet "
        while True:
            action = self.act(total_text)
            if action in self.action_to_index:
                index_action = self.action_to_index[action]
                total_text += " " + action + ".\n"
                print(total_text)
                break
        
        self.trial += 1

        return one_hot(np.array([index_action]), 2)[0]

    


if __name__ == '__main__':
    gpt = GPT_ENV(0.0, 4, 0)
    a = gpt.reset()

    k = 0
    for i in range(200):
        r = 0
        if a == 1:
            if np.random.uniform(0, 1) < 0.5:
                r = 1

        if a == 0:
            if np.random.uniform(0, 1) < 0.1:
                r = 1

        a = ql.step(r)
        k += a

        print(a)

    print(k / 200)