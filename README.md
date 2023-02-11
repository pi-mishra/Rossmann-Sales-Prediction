# Rossmann-Sales-Prediction (Capstone project)

## Table of Content
  * [Abstract](#abstract)
  * [Problem Statement](#problem-statement)
  * [Data Description](#data-description)
  * [Project Outline](#project-outline)
    - 1 [Data Wrangling](#data-wrangling)
    - 2 [Normalization](#normalization)
    - 3 [EDA](#eda)
    - 4 [ Encoding categorical values](#encoding-categorical-values)
    - 5 [Feature Selection](#feature-selection)
    - 6 [Model Fitting](#model-fitting)
    - 8 [Hyper-parameter Tuning](#hyper-parameter-tuning)
    - 9 [Metrics Evaluation](#metrics-evaluation)
  * [Conclusion](#conclusion)
  * [Reference](#reference)


# Abstract
This project focuses on forecasting daily sales for over 3,000 drug stores of Rossmann in 7 European countries. Historical sales data for 1,115 stores and supplemental information about the stores were provided. The sales were influenced by various factors such as promotions, competition, school and state holidays, seasonality, and locality. A machine learning model was developed to predict the daily sales and improve the efficiency of the store managers in their sales forecasting. The model was trained on the historical sales data and store information, including features such as store Id, store type, promotions, competition distance, and more. The results of the model showed promising accuracy in predicting the daily sales for the Rossmann stores.

# Problem Statement
The problem statement is to predict the daily sales for Rossmann stores based on historical sales data and various influencing factors such as promotions, competition, holidays, seasonality, and store location. The goal is to provide store managers with accurate sales predictions for up to six weeks in advance to help them make informed decisions. The task is to forecast the sales column for a test set of 1,115 Rossmann stores using data such as store Id, store type, assortment level, competition distance, promo information, and other relevant factors. The accuracy of the predictions can vary based on individual store circumstances, and the challenge is to develop a model that can effectively consider all the influencing factors to produce accurate sales predictions.

# Data Description
We have a 2 dataset which contains information about the sales data of 1115 stores after merging the data we have 1017209 rows and 18 columns. Sharing the deatails of the columns below-


   
Conclusion and recommendation-
   •	Store type B has the maximum number of average sales. 
   •	Maximum number of stores are closed on Sunday. So there is increase in average sales on Monday.
   •	Assortment type B has the maximum number of average sales.
   •	17.9% of data suggest that closure of public-school effect the sales.
   •	Average competition distance is 5.4.
   •	Sales in 2013, 2014 and 2015 are stagnant, there is a hardly increase in sales.
   •	As the competition distance increase the sales decrease. This shows that customers are churning to Rossmann Stores, so we should open more store where competition       is available.
   •	Store type B should be increased and assortment B should be added to it.
   •	When there is closure of public school more promotion should be done to increase the sales.
   •	Stores should be open on Sundays as there is demands on Sundays also, some people might have gone to the competition stores for the products.
   •	If there was use of promo the sales increased, use of promo should be in increased in the stores specially during holidays.
   •	Decision tree should be used in the analysis as it outperforms other models and has a r-squared score of 0.97.



