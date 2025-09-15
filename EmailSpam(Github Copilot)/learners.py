from domiknows.sensor.pytorch.learners import LinearLearner

# Learners for Model 1 and Model 2
model1_learner = LinearLearner(input_size=128, output_size=2)  # Spam/Legitimate
model2_learner = LinearLearner(input_size=128, output_size=2)