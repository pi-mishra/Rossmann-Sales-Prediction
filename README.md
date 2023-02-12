# Rossmann-Sales-Prediction (Capstone project)

## Table of Content
  * [Abstract](#abstract)
  * [Problem Statement](#problem-statement)
  * [Data Description](#data-description)
  * [Project Outline](#project-outline)
    - 1 [Data Wrangling](##data-wrangling)
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

| Feature Name | Type | Description |
|----|----|----|
Store | int64 | a unique Id for each store
Sales | int64 | the turnover for any given day (this is what you are predicting)
Customers | int64 | the number of customers on a given day
Open | int64 | an indicator for whether the store was open: 0 = closed, 1 = open
StateHoliday | object | indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
SchoolHoliday | int64 | indicates if the (Store, Date) was affected by the closure of public schools
StoreType | object | differentiates between 4 different store models: a, b, c, d
Assortment | object | describes an assortment level: a = basic, b = extra, c = extended
CompetitionDistance | float64 | distance in meters to the nearest competitor store
CompetitionOpenSince[Month/Year] | float64 | gives the approximate year and month of the time the nearest competitor was opened
Promo | int64 | indicates whether a store is running a promo on that day
Promo2 | int64 | Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
Promo2Since[Year/Week] | float64 | describes the year and calendar week when the store started participating in Promo2
PromoInterval | object | describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store


# Project Outline

##1. Data Wrangling - After loading our dataset we used .isnull().mean() to get the percentage of null data in the columns. We dropped promo2sinceweek , promo2sinceyear, promointerval as there are 49% data null. Considering date as categories we have filled ‘CompetitionOpenSinceMonth’ and 'CompetitionOpenSinceYear' with mode. Considering there is no competition or the competion is so far that there is no account of the data, so filling null values with 0.There were 0 sales when the stores were closed so we are dropping the ‘Open’ column as it will not provide any useful information for the analysis.
![r1](https://user-images.githubusercontent.com/102457813/218310603-284aa0fd-2cbe-48b2-9bcf-290f7083160e.png)


   
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



