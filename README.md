# PREDICTING TOTAL SCORE OF IPL MATCH

We Build a model on the “total” column, of IPL data set using a RandomForestRegressor Algorithm

Data set Containing following features.

       mid,date,venue,bat_team,bowl_team,batsman,bowler runs,wickets,overs,runs_last_5,wickets_last_5,striker,non-striker,total 
       
Step to Build Machine Learning Model

     1. Import the dataset, and remove the unwanted columns
     
     2. Follow these 4 steps of machine learning workflow and make sure that the all rules of modelling are satisfied(before 3rd step)
        Step of Machine Learning.
                   1. Extract
                   2. Split the dataset into training and testing dataset
                   3. Train the model on training data
                   4. Test the model on testing data
                   
        Rule rules of modelling to Satisfy Before Train model
                   1. Features and target should not have any null values.
                   2. Features should be of the type array/ dataframe.
                   3. Features should be in the form of rows and columns
                   4. Features should be on the same scale
                   5. Features should be numeric
     3. We can predict on a new dataset only containing the features 
     4.        

