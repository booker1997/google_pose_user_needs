from ergonomics.reba import RebaScore
import requests
import numpy as np
import json

url = 'https://www.dropbox.com/s/e72lwn9n9gkoygj/ergonomics.json?dl=1'
r = requests.get(url)
sample_pose = np.array(r.json()['sample_pose'])
print(sample_pose,sample_pose.shape)

rebaScore = RebaScore()

body_params = rebaScore.get_body_angles_from_pose_right(sample_pose)
arms_params = rebaScore.get_arms_angles_from_pose_right(sample_pose)

rebaScore.set_body(body_params)
score_a, partial_a = rebaScore.compute_score_a()

rebaScore.set_arms(arms_params)
score_b, partial_b = rebaScore.compute_score_b()

score_c, caption = rebaScore.compute_score_c(score_a, score_b)

print("Score A: ", score_a, "Partial: ", partial_a)
print("Score A: ", score_b, "Partial: ", partial_b)
print("Score C: ", score_c, caption)