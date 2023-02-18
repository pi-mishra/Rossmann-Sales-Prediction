# Rossmann-Sales-Prediction (Capstone project)

## Table of Content
  * [Summary](#summary)
  * [Problem Statement](#problem-statement)
  * [Data Description](#data-description)
  * [Project Outline](#project-outline)
    - 1 [Knowing the data](##Knowing-the-data)
    - 2 [Normalization](#normalization)
    - 3 [EDA](#eda)
    - 4 [ Encoding categorical values](#encoding-categorical-values)
    - 5 [Feature Selection](#feature-selection)
    - 6 [Model Fitting](#model-fitting)
    - 8 [Hyper-parameter Tuning](#hyper-parameter-tuning)
    - 9 [Metrics Evaluation](#metrics-evaluation)
  * [Conclusion](#conclusion)
  * [Reference](#reference)


# Summary
This project involved merging two datasets into one dataframe called all_store, which had 1017209 rows and 18 columns. The dataframe consisted of int, object, and float datatypes, and object datatypes were later converted to int. There were no duplicate values, but six columns had null values that needed to be treated.

To better understand the data the year, month, and day from the "date" column were extracted and then dropped. The "Open" column was also dropped since no sales occurred when stores were closed. Furthermore, 49% of data was missing from columns "Promo2SinceWeek," "Promo2SinceYear," and "PromoInterval," so they were dropped. Considering date as categories we have filled ‘CompetitionOpenSinceMonth’ and 'CompetitionOpenSinceYear' with mode. Considering there is no competition or the competion is so far that there is no account of the data, so filling null values with 0.

The most common type of store was "a" with 551,627, followed by "d" with 312,912, "c" with 136,840, and "b" with 15,830. It was discovered that store type and sales were correlated, with more stores leading to more sales. Monday had the highest number of sales, followed by Tuesday, Friday, Wednesday, Thursday, Saturday, and Sunday. Similarly, average sales for different days of the week followed the same trend. Store B had higher average sales than other stores C, A, and D, respectively. Monthly sales over the year increased, with an increase in the number of stores leading to more customers' visits and higher sales. Sales data were right-skewed, and 17% of stores were affected by the closure of public schools. No promos were run on Saturday and Sunday, and sales increased when the promo was used.

It was observed that the "CompetitionDistance" column had a large positive skewness of 2.93, indicating a heavily skewed right-tailed distribution. On the other hand, "Sales" had a small positive skewness of 0.64, indicating a slightly skewed right-tailed distribution, while "Customers" had a moderate positive skewness of 1.60, indicating a skewed right-tailed distribution.

To remove outliers the IQR method was used and then one-hot encoding was performed. The data were then scaled using MinMaxScaler.

Conclusion was made that the decision tree and XG boost models outperformed linear, ridge, and lasso regression models in predicting sales of Rossman stores. The decision tree model achieved an R2 score of 0.973 and a low RMSE of 538, while the XG boost model achieved an R2 score of 0.926 and a relatively low RMSE of 849. In contrast, the linear, ridge, and lasso regression models achieved similar R2 scores of around 0.893, with higher RMSE values ranging from 1021 to 1022. The decision tree and XG boost models are more accurate and may be more robust to non-linear relationships in the data, making them the preferred models for predicting sales of Rossman stores.

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

## Knowing the data-
   * The 2 datasets are merged into 1 dataframe i.e. all_store consisting of 1017209 rows and 18 columns.
   * The dataframe consist of int, object and float datatypes. Object data type will be changed to int later.
   * There are 0 dublicate value.
   * There are 6 columns (CompetitionDistance, CompetitionOpenSinceMonth, CompetitionOpenSinceYear, Promo2SinceWeek, Promo2SinceYear, PromoInterval) that has null          values it needs to be treated.


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



