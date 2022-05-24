# Prediction-of-economic-activity-6th-place-solution
Solving the hackathon problem from Tochka Bank

This repository contains all files to train classifier of economic activity. This repository does not include any models.

## Pipeline

The pipeline is made up of several blocks:
![[blocks](https://github.com/RadmirZ/Prediction-of-economic-activity-6th-place-solution/blob/main/catboost%20pipeline.jpeg)](https://github.com/RadmirZ/Prediction-of-economic-activity-6th-place-solution/blob/main/pipe_scheme.jpeg)

And a step of post-processing:
* selection of the model's prediction threshold

## Feature Engineering

Most of all features in my solution are based on time periods, month, season of the year, week(didn't work). The main boost was gain by aggregation one of time period and other category column from the dataframe. Aggregations was the same all the time ( 'mean', 'median', 'std', 'min', 'max', 'sum', 'len' ). Normalization features by user (divide num of unique operations by total count) for given clieint worked good too. Extracting features from *contractor_id* was inclined to overfitting, in my case it was overfitting to local validation, which was really great and correlated very confidently with the leaderboard. So, in my final solution I didn't use *contractor_id* for training. Finally, for this task we were able to extract embeddings with *PyTorch Lifestream* algorithm. As of my experience, It could work very well and could get high scores, but it is also inclined to overfitting.

## Model and its parameters

Well, the most used model for this task was catboost and that was the only model I've trained. However, as of my experience there are a lot of ways for modeling, such as 1D-CNN, GRU/LSTM networks, LightAutoML and other similar models. But I settled on catboost in order to work more on data and features, since this competition is completely about features, not about models. I manually selected the parameters for the model and it gave a solid boost. The rule I followed was "smaller lr, more epochs" (It gives you +0.005 to your score). As well, post-processing was the significant part of the score growth. I sorted the model's prediction threshold to get better score, It was one more thing that could lead to overfitting.




