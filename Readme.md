-Aim: This project deals with the Sentiment analysis of IMDB's movie reviews. It comares various algorithms and compares their confidence and accuracy.

Structure:
The structure is as following:
	1. short_reviews: It is the collected data-set. The data set can also be found at https://ai.stanford.edu/~amaas/data/sentiment/. Althought we are not using the whole dataset for our purpose because of machine limitations.
	2. Stanford-postagger. It is a jar file that is used for tagging parts of speech.
	3. data_parsing.py: It is where the whole code for generating the models and testing and getting the accuracy is. It has data cleaning and pre-processing steps included. Preprocessing includes tokenization, stemming, etc.
	4. 	trained_models: It contains all the trained models and can be directly used for testing purposes by unpickling the dump.
		
		