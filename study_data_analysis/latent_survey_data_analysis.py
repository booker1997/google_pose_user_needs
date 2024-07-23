import pandas as pd
import numpy as np



VIDEO = 'vacuum'
# VIDEO = 'desk'

if VIDEO == 'vacuum':
    responses = pd.read_csv('data/responses_vacuum.csv')
    final_data_path = 'data/latent_labeled_needs_vacuum.csv'
else:
    responses = pd.read_csv('data/responses_desk.csv')
    final_data_path = 'data/latent_labeled_needs_desk.csv'

names = []
needs = []
latents = []
question_answers = []
for participant in responses['name'].unique():
    # print(responses[responses['name']==participant])
    part_df = responses[responses['name']==participant]
    for i,need in enumerate(part_df['phrase']):
        new_answers = np.zeros(4)
        print(i,participant,need,part_df['impactful'].to_numpy()[i])
        if part_df['impactful'].to_numpy()[i] == 'Strongly agree' or part_df['impactful'].to_numpy()[i] == 'Agree':
            new_answers[0] = 1
            if part_df['implicit'].to_numpy()[i] == 'Strongly agree' or part_df['implicit'].to_numpy()[i] == 'Agree':
                new_answers[1] = 1
                if part_df['obvious'].to_numpy()[i] == 'Strongly disagree' or part_df['obvious'].to_numpy()[i] == 'Disagree':
                    new_answers[2] = 1
                    if part_df['inefficient'].to_numpy()[i] == 'Strongly disagree' or part_df['inefficient'].to_numpy()[i] == 'Disagree':
                        new_answers[3] = 1
                        latent = 1
                    else:
                        latent = 0
                else:
                    latent = 0
            else:
                latent = 0
        else:
            latent = 0
        names.append(participant)
        needs.append(need)
        latents.append(latent)
        question_answers.append(new_answers)



final_df = pd.DataFrame()
final_df['name'] = names
final_df['need'] = needs
final_df['latent'] = latents
final_df['question_answers'] = question_answers        

print(final_df)
final_df.to_csv(final_data_path)