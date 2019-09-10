# collaborative_filtering_recommendation_system
Using collaborative filtering to implement user based and item based recommendation system.


# Train
To fit and test recommendation system, run:

 `python main.py`
 
 
# Data
Movie Lens Small Latest Dataset

users count: 610

movies count: 9742 

ratings count: 100836 


To download:

https://www.kaggle.com/shubhammehta21/movie-lens-small-latest-dataset


To use custom data:

Update `data_path` to custom data file path, and set `user_label`, `item_label`, `score_label`. For large sparse dataset, the fitting process would be really slow. You can set `min_user_count` and `min_item_count` in `load_data` to reduce data and speed up the test.

