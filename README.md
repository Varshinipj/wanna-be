# wanna-be - Fashion Recommendation System
A hybrid recommendation system built to support black owned small businesses. Recommendations and links to the sites will be suggested.


## Authors
  - Varshini P J
  - Dharineesh Karthikeyan


## Project Details
👗 This project is based on building a content-based Fashion Recommendation
System.

👗 Building a recommendation system needs two datasets - the training and the
mapping dataset. The training dataset is the DeepFashion dataset (which is
available only for non-commercial purposes). The mapping dataset is created
using Webscraping using BeautifulSoup in Python.

👗 Scraping : https://www.kaggle.com/code/varshinipj/scraping-data-and-images-from-a-shopping-site

👗 Training dataset: Category and Attribute prediction benchmark dataset from DeepFashion

👗 Mapping dataset: A local database created by scraping images and pricing data from various black-owned small businesses.

👗 Model used for attribute prediction: FastAI - attribute prediction stage 1 ResNet34 

👗 Evaluation metric: FBeta score

👗 Loss Function: Label Smoothing BCE with Logits Loss Function

👗 Color Prediction:  using KMeans algorithm

👗 The scores for attributes and color are stored in a dataframe, whenever a picture is uploaded onto the recommender, the image/product with the highest score will
be recommended first.

👗 The UI is built using Streamlit.

👗 Working final product: https://github.com/Varshinipj/wanna-be/blob/main/streamlit-app-2022-06-06-02-06-16.mp4


https://github.com/user-attachments/assets/d4c121c8-3db0-4852-af05-7a0ced0fc798

