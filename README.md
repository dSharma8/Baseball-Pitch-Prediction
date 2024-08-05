# Baseball-Pitch-Prediction

Final project for CS 4641. Created alongside Pau Sum and Jerolyn Prema.

## Introduction/Background
Baseball is one of the most popular sports in the United States, often referred to as “America’s pastime.” With Major League Baseball’s total revenue consistently increasing [1] and the betting industry reaching a revenue of $7 billion in 2023 [2], data analytics has become increasingly important. The MLB collects extensive statistics on baseball performance metrics for teams and players. In particular, the analysis of pitch trajectories is crucial for training pitchers and batters alike. Pitch data can also provide additional content for viewers in the form of 3D pitch trajectory visualizations in between lulls of the game [3]. This project will utilize data sourced from MLB’s Statcast Search [4] – focusing on data from the Atlanta Braves 2018 MLB season.

## Problem Definition
**Problem:** In baseball, numerous factors contribute to winning a game, and various statistics can predict the success of both teams and individual players. Pitching is arguably the most critical aspect of baseball, as every subsequent game action depends on the outcome of each pitch [5]. 

As such, the original focus of this project was to focus on pitching, using pitch data such as pitch outcome, type, speed, and movement to predict the likelihood of a strike. The goal was that this information be used to train both pitchers and batters. However, our problem definition has since then shifted due to the immense complexity of baseball pitching strategy, and the large number of factors that can affect whether or not a specific pitch results in a strike. Everything from how many players are on base, to the level of exhaustion of a pitcher/batter as the game progresses, to the current number of balls/strikes/outs, to which types of pitches have been thrown by a pitcher in a specific inning can all affect the strategy and performance of both pitchers and batters. Not all of these features were supported enough in the data we sourced [4].

Given our new understanding of both our data and of baseball pitching strategy, we decided to shift our model to determining the type of a given pitch based on its measured characteristics. This still accomplishes the original goal of providing resources to help train both pitchers and batters. 

The analysis of pitching trajectories is a centerpiece of baseball strategy in many professional baseball leagues, including Major League Baseball (MLB) and the Korean Baseball Organization (KBO) [3]. It is “critically important” for teams to be able to determine the type of pitch based on its trajectory and other characteristics [6] so that a pitcher might examine and thus improve their pitching. Yet this information must be determined manually by specialists [6] in a time consuming process. Our project attempts to use various machine learning algorithms to categorize a pitch based on its physical characteristics including speed, rotation, and movement.

**Motivation:** Understanding and predicting pitch outcomes can significantly enhance training methods and strategies in baseball. By efficiently and accurately determining the type of a certain pitch based on its characteristics, teams are better able to compare prospective pitchers, pitchers are better able to analyze and improve their own performance, and even opposing batters are better able to prepare for games. This can improve overall performance and lead to more exciting games.  

## Methods
To predict the likelihood of a strike using pitch data from MLB’s Statcast Search, in our models we applied the following comprehensive data preprocessing techniques and machine learning algorithms.
### Data Preprocessing Methods
**Feature Scaling:** 'StandardScaler' from scikit-learn.
Pitch data includes various features like speed, movement, spin rate, etc that are on different scales. The ranges of all these features should be standardized so that each feature is able to contribute approximately proportionately to the final outcome. This is why we standardized these features as part of the data preprocessing portion of our model.

**Encoding Categorical Features:** 'OneHotEncoder' from scikit-learn 
Our pitch data includes categorical features like pitch outcome (StikeCalled, StrikeSwinging, Ball, etc.) or pitcher handedness (left-handed or right-handed). In order to feed these to our machine learning algorithm, we had to first transform them into numerical features using a OneHotEncoder.

**Other:**
On top of these methods, we can do some initial analysis to remove certain features that are clearly not relevant for our model. Firstly, we discarded any data points that had any null values (including data points for which the pitch type was “other” or “undefined”). Additionally, since we are focusing on the characteristics of each individual pitch and not on any specific players/game, it is clear that features such as Pitcher_ID, Batter_ID, Game_Date, and Inning are all features that are irrelevant to our model. We can also exclude any features that related to how many balls, outs, and strikes that were already thrown in a given inning.

### Machine Learning Algorithms
**Logistic Regression:** 'LogisticRegression' from scikit-learn 
Logistic Regression is a linear model that works by taking in the input as independent variables and produces a probability value for each output (which all together sum to 1). Logistic Regression is a straightforward method and is able to effectively handle the multiple features of our data. Additionally, Logistic Regression can also tell us which features of our data are the most important in predicting the pitch type, allowing for simple and straightforward analysis. 

**Random Forest Classifier:** 'Random Forest Classifier' from scikit-learn
The random forest classifier is built using the decision tree algorithm. In a decision tree, internal nodes represent features, each branch represents a decision, and each leaf node represents a result.	The main issue with decision trees is that they are very sensitive to the training data and are prone to overfitting. The random forest classifier minimizes overfitting by selecting random subsets of the original training data and creating a decision tree for each. This process introduces more randomness and helps in making the trees less correlated with each other. Each tree in the forest makes a class prediction, and the class with the most votes is the final prediction of the Random Forest

**Support Vector Machine:** 'Support Vector Machine' from scikit-learn
Support Vector Machines (SVMs) are supervised learning methods used for classification, regression and outlier detection. SVMs are very effective in high dimensional spaces. SVMs are also versatile because of their ability to handle different relationships based on the kernel function. A SVM is traditionally binary, but can be expanded to support multiclass classification via the “one-v-rest” or “one-v-one” methods. In “one-v-rest” we train a binary classifier for each class, distinguishing between that class and the rest and we take the classifier with the highest confidence score. In “one-v-one” we train a binary classifier for every pair of classes and then compare them. For our model, we used “one-v-one” because while it is computationally more expensive, it has the potential to be more accurate since it can consider interactions between classes. For our pitch prediction problem, the SVM would try to find the decision boundaries between each type of pitch based on the features of the pitch. Since SVMs are effective with handling high dimensional spaces, SVM should excel at handling the high dimensional feature space created by pitch speed, movement, spin rate, etc. The use of various kernel functions also allows us to create a more complex decision boundary, which would be created by a high dimensional prediction problem like ours.

## Results/Analysis
### Logistic Regression Model
The logistic regression model achieved an accuracy of approximately 88.52% on the test set. This indicates that the model performs relatively well in classifying pitch types based on the given features. The feature importance analysis reveals that release_speed, z_movement, and release_pos_x are among the most significant predictors.

**Strengths:** The model provides clear interpretability through feature coefficients, allowing us to understand the importance of each feature. The high accuracy and AUC scores indicate that the model can effectively classify pitch types.

**Weaknesses:** Certain pitch types may still be misclassified, as indicated by the off-diagonal elements in the confusion matrix. The model might be overfitting to specific features, as seen in the relatively high coefficients for some features.

### Random Forest Classifier
The feature importance analysis reveals which features contribute most to the model's predictions. Features such as release_speed, z_movement, and release_pos_x were among the most significant predictors.

**Strengths:** The Random Forest model is robust and can handle a large number of features without overfitting. It provides feature importance metrics, which help in understanding the contribution of each feature to the model's predictions. The model achieved high accuracy, indicating its effectiveness in classifying pitch types based on physical characteristics.

**Weaknesses:** The Random Forest model may require more computational resources and time to train compared to simpler models. While it provides feature importance, it does not offer the same level of interpretability as logistic regression in terms of understanding the relationships between features and outcomes.

### Support Vector Machine
The SVM model achieved an accuracy of approximately 95.17% when using the RBF (radial basis function) kernel function. This kernel function (which is also the default kernel for sklearn’s support vector classifier) performed best when compared with other kernel functions such as linear (90.99%), polynomial (94.30%), and sigmoid (74.15%), which is why it was chosen as the final kernel function for this model. This shows that the model was very accurate when classifying a pitch’s type based on its physical characteristics. However, we are not able to determine which features contribute more or less to the model since the weights assigned to each feature is only available when we use the linear kernel function (which we did not end up using). This is because in the other kernels, the separating plane exists in another dimension and thus its coefficients are not directly related to the input space.

**Strengths:** SVMs are very good at handling high dimensional problems, including this one, as can be seen by its high accuracy score. A testing accuracy of 95.17% means that this model is very good at determining pitch type based on physical characteristics such as speed and movement.  

**Weaknesses:** One weakness of the SVM is that we do not have access to feature importance weights. While this does not affect the model or its accuracy, it does give us less information to analyze and can make it harder to improve the model. Another disadvantage is that it can be very computationally expensive, especially when doing multiclass classification with the “one-v-one” method as for k classes we need to compute (k * (k - 1) / 2) binary classifiers. 

## References
[1] C. Gough, “MLB league revenue 2024,” Statista, https://www.statista.com/statistics/193466/total-league-revenue-of-the-mlb-since-2005/   

[2] “US Sports Betting Revenue Tracker - how much revenue is each state generating in 2023?,” Oddspedia, https://oddspedia.com/us/betting/sports-betting-revenue 

[3] H. Lee, J. Kim, J. Kim, J. Yu and W. -Y. Kim, "Start-End Time Detection in Baseball Videos for Automatic Pitching Trajectory Analysis," 2019 International Conference on Electronics, Information, and Communication (ICEIC), Auckland, New Zealand, 2019, pp. 1-4, doi: 10.23919/ELINFOCOM.2019.8706498.  

[4] “Statcast Search,” baseballsavant.mlb.com, https://baseballsavant.mlb.com/statcast_search 

[5] H. Lee, J. Kim, J. Kim and W. -Y. Kim, "A Method of Measuring Baseball Position at the Strike Zone," 2020 International Conference on Electronics, Information, and Communication (ICEIC), Barcelona, Spain, 2020, pp. 1-3, doi: 10.1109/ICEIC49074.2020.9051039.

[6] J. Schuh and L. Kong, "Classifying Pitch Types in Baseball Using Machine Learning Algorithms," 2023 IEEE Asia-Pacific Conference on Computer Science and Data Engineering (CSDE), Nadi, Fiji, 2023, pp. 1-6, doi: 10.1109/CSDE59766.2023.10487702.

[7] D. Adler, “Identifying Pitch Types: A fan’s guide,” MLB.com, https://www.mlb.com/news/identifying-pitch-types-a-fan-s-guide. 
