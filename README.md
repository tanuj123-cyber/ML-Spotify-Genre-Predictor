# ML-Spotify-Genre-Predictor

This project aimed to train an adaBoost classification model over a dataset of audio features of approximately 50k songs retrieved from Spotify in order to evaluate its performance as a genre predictor. 
We clean the data by using one-hot encoding and numerical mapping for string data types, accounting for empty values, and perfoming dimension reduction technique LDA due to the significant size of our dataset. 
After data cleaning and training, we engineer the adaBoost model to predict the genre of a song given other audio features, with an AUROC score of approximately 0.876. 

Refer to the PDF file for the report that was written, outlining the specific details regarding the data cleaning and model performance. 
