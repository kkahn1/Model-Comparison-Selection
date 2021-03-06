#' ---
#' title: "Machine Learning Tutorial: Model Comparison and Selection"
#' author: "Kayla Kahn"
#' date: "03/08/2021"
#' output: html_document
#' ---
#' 
#' The data are Extended Data on Terrorist Groups (EDTG) from Hou, Gaibulloev, 
#' and Sandler (2020). These data are an extension of Jones and Libicki's 2008 
#' data on terrorist group aspects and are connected to the Global Terrorism 
#' Database. They are group-year panel data with information on groups such as 
#' goals, orientation, total attacks, total casualties, etc. For today, I've 
#' collapsed the data to be cross sectional using the replication code, because 
#' this is a difference of dealing with 760 observations vs over 9000. Note that 
#' if you google EDTG, you will find the main dataset that has only group level 
#' covariates, but the data being used today is from their replication file for 
#' their 2020 paper and includes country covariates based on group base(s).
#' 
#' The outcome of interest is whether a group has ended. I'll be using neural 
#' networks and show how to compare both within the training as well as comparing 
#' a few of the best ones once you have chosen which to use.
#' 
## ----------------------------------------------------------------------------------------------
library(nnet)
library(caret)
library(pROC) 
library(ROCR) 

#' 
#' 
#' We'll read in the data and we can see that the outcome is not terribly unbalanced. 
#' Then we will split into train and test sets.
## ----------------------------------------------------------------------------------------------
mldata<-read.csv("https://raw.githubusercontent.com/kkahn1/Model-Comparison-Selection/main/edtgcs.csv",stringsAsFactors=TRUE)
length(which(mldata$end==1))
# little bit of data cleaning
mldata$competitor<-log(mldata$competitor+1) # log competitor as they did
mldata$y_start<-NULL # lots of missing data and we won't be using this variable
mldata$y_end<-NULL # lots of missing data and we won't be using this variable
mldata$lnpeaksize<-NULL # lots of missing data and we won't be using this variable
mldata<-na.omit(mldata)
# since this is binary, set it to a factor with names or else it will yell at you
mldata$end<-ifelse(mldata$end==1,"end","active")
mldata$end<-factor(mldata$end, levels = c("end","active"))
levels(mldata$end)
set.seed(9986000)
# split
partition<-createDataPartition(mldata$end, #outcomes 
                               p=.75, #percent of data for training
                               list=FALSE) #keep as matrix
traindf<-mldata[partition,] #use which of the df in the .75 partition
testdf<-mldata[-partition,] #use the opposite (the rest of the data)

#' 
#' 
#' 
#' For creating the models we are going to use nnet within caret. nnet is fairly 
#' simple in that it only has one hidden layer so for the best fitting model, 
#' you may want a different package. For nnet we set two hyperparameters: size 
#' which is the number of neurons in the hidden layer, and decay which prevents 
#' overfitting. You can instruct caret to search a range of hyperparameters and 
#' it will try each combination. Here is an example of setting the hyperparameters. 
#' This says try 1-5 and 10 neurons in the hidden layer and try these 5 levels of decay.
## ----------------------------------------------------------------------------------------------
grid1<-expand.grid(size = c(1:5,10),decay = c(0, 0.05, 0.1, 1, 2))

#' 
#' We also set the "control" and this is where you tell how to compare between 
#' the different models in the training. The default is bootstrapping but it can 
#' do several ways of validating. I'm going to show you a few with cross-validation. 
#' Here's an example of setting it to cross validation and using 5 folds. This tells 
#' it to do cross validation with 5 folds. savePredictions=TRUE will let use see the 
#' predictions with this cross validation "test" set.
#' 
## ----------------------------------------------------------------------------------------------
control1<-trainControl("cv",number=5,savePredictions=TRUE)

#' 
#' Caret will use all the combinations of the size and decay hyperparameters. 
#' For cross validation in general, it will split the training set into training 
#' and validation sets, so it will run the model on the training set and fit on 
#' the validation set. 
#' 
#' For k-fold cross validation, which we will do below, it will split it into 
#' k sets (non overlapping). Then it will train the model by leaving out one 
#' of the validation sets, predict on the validation set (These are not your 
#' final predictions. Your test set from before hasn't been touched yet.), and 
#' will repeat this so that each validation set is left out once. For what we 
#' have set, it is saying to split the training data into 5, and each time leave 
#' one of the 5 out as the cross validation "test" set. It will compare the models 
#' based on a metric that you can set accuracy, kappa, ROC, etc and choose the best 
#' model. 
#' Here is a good post that explains the different metrics. 
#' https://machinelearningmastery.com/machine-learning-evaluation-metrics-in-r/
#' 
#' Let's train the first model. We will use the tuning and control that we set above
## ----------------------------------------------------------------------------------------------
formula1<-as.formula(paste("end~",paste(c("shr_trans","diversity","mul_bases",
                                          "lpop","ly","total_atks","nonterr_deaths",
                                          "competitor", "polity2","polity2sqr","duration",
                                          "dur2"),collapse ="+")))

set.seed(9986000)
model1<-caret::train(formula1,data=traindf, #formula and data
                     metric="Accuracy",#how to choose the best model
                     method="nnet",#type of model
                     tuneGrid=grid1,trControl=control1,#see above
                     maxit=300, #allow more iterations for it to converge
                     trace=FALSE)#don't print out everything as it runs
#We can print model1$pred to see the predictions across the folds with each size and decay
head(model1$pred,10)
#We can pull up the summary where it has chosen the best model from each 
#combination of the size and decay
#The final model is the best best
model1

#' We can see from the summary that it ran each of 6 sizes at each of 5 decay 
#' parameters, and we know from model1$pred that it did this process 5 different 
#' times for the five folds. It tells us which model it choose. We can look at 
#' the accuracy (in cross validation) and the kappa. With Kappa 1 is perfect 
#' agreement and 0 is choosing at random. So we know from the cross validation 
#' (but not actual predictions) that the model has a pretty high accuracy and 
#' is also performing better than as if random.
#' 
#' Another type of cross validation is leave one out cross validation. It is 
#' like kfold cross validation but for every instance in the dataset. It will 
#' leave out an instance as the cross validation "test" and train the model on 
#' the other data to choose the best final model. This will take a long time and 
#' probably isn't feasible, but here is some code for if you want to do it that way.
#' 
## ----------------------------------------------------------------------------------------------
trainControl("LOOCV",savePredictions=TRUE)

#' 
#' Now we'll run a couple more models and then compare. I'll use a slightly 
#' different specification as well as show a different way of cross validating. 
#' Then at the end we can compare the predictions of each. This time we are going 
#' to do k fold cross validation with repeated sampling. This improves on k fold 
#' cross validation because with k fold cross validation, each split of the data 
#' may be very different. With repeated k-fold cross validation it will split the 
#' data k times but repeat the process x times. 5 fold cross validation with 3 
#' repeats will split the data into 5 folds and cross validate and will do the 
#' whole process 3 different times - so 3 different ways of splitting into 5 folds. 
#' 3, 5, and 10 are common values for the repeats. The more folds and repeats you 
#' have, the longer it will take, especially if you are searching a range of 
#' hyperparameters, because it will then do this for the combinations of 
#' hyperparameters. I have done some investigating beforehand to set the them 
#' so that it doesn't take as long during the tutorial.
#' 
## ----------------------------------------------------------------------------------------------
# this is the same as the last formula but removes the duration variables
formula2<-as.formula(paste("end~",paste(c("shr_trans","diversity","mul_bases",
                                          "lpop","ly","total_atks","nonterr_deaths",
                                          "competitor", "polity2","polity2sqr"),collapse ="+")))

#set seed again because I don't trust r
set.seed(9986000)

grid2<-expand.grid(size = c(3,4,5),decay = c(.3,.5,.8,1))
control2<-trainControl("repeatedcv",number=5,repeats=3,savePredictions=TRUE)

model2<-caret::train(formula2,data=traindf, #formula and data
                     metric="Accuracy",#how to choose the best model
                     method="nnet",#type of model
                     tuneGrid=grid2,trControl=control2,
                     maxit=500, #allow more iterations for it to converge
                     trace=TRUE)# this time we can watch it run
#see the predictions across the folds with each size and decay
head(model2$pred,10)
#We can pull up the summary where it has chosen the best model from each 
#combination of the size and decay
model2

#' We can see that accuracy is a little less than before but kappa is worse. 
#' 
#' 
#' This time the model will include the log of elevation variable and remove the 
#' proportion of attacks that are transnational. We will do the k fold cross validation 
#' that we did in model 1.
#' 
## ----------------------------------------------------------------------------------------------
formula3<-as.formula(paste("end~",paste(c("diversity","mul_bases","lpop","ly","total_atks","nonterr_deaths","polity2","polity2sqr","duration","dur2","log(elevation)","competitor"),collapse ="+")))

set.seed(9986000)
grid3<-expand.grid(size = c(1:5),decay = c(.4,.5,.6,.7))
model3<-caret::train(formula3,data=traindf, #formula from before
                     metric="Accuracy",#how to choose the best model
                     method="nnet",#type of model
                     tuneGrid=grid3,trControl=control1,#new grid, control from before
                     maxit=400, #allow more iterations for it to converge
                     trace=FALSE)#don't print out everything as it runs

#We can pull up the summary where it has chosen the best model

#' 
#' 
#' Now we have three models. Now we will test how these models work on our test set. 
## ----------------------------------------------------------------------------------------------
#model1
preds1<-predict(model1,newdata=testdf,type="raw")
confmat1<-confusionMatrix(data=preds1,reference=testdf$end)
#model2
preds2<-predict(model2,newdata=testdf,type="raw")
confmat2<-confusionMatrix(data=preds2,reference=testdf$end)
#model3
preds3<-predict(model3,newdata=testdf,type="raw")
confmat3<-confusionMatrix(data=preds3,reference=testdf$end)
#use this to see the formulas
?caret::confusionMatrix

#' 
#' We can also look at the ROC curve. We'll use the package pROC but there are 
#' other packages that can do this including ROCR.
## ----------------------------------------------------------------------------------------------
# first do the predictions as probabilities
pp1<-predict(model1,newdata=testdf,type="prob")
pp2<-predict(model2,newdata=testdf,type="prob")
pp3<-predict(model3,newdata=testdf,type="prob")

# my target is "end" but I also have a level of it as "end"--
# when we say testdf$end we are referring to the variable 
# when pp1$end, we are referring to the predicted probs for "end" within pp1
# as opposed to predicted probs for "active"
roc1<-roc(testdf$end,pp1$end)
roc2<-roc(testdf$end,pp2$end)
roc3<-roc(testdf$end,pp3$end)

# best plots the threshold with the highest sum sensitivity + specificity
# youden says max(sensitivities + specificities) is the optimal threshold
plot(roc1,print.thres="best",print.thres.best.method="youden")
plot(roc2,print.thres="best",print.thres.best.method="youden")
plot(roc3,print.thres="best",print.thres.best.method="youden")

#We can look at the area under the curve
auc(roc1)
auc(roc2)
auc(roc3)

# then we can save the coordinates to look at the threshold and accuracy
roccoord1<-coords(roc1,"best",best.method="youden", ret="all")
roccoord2<-coords(roc2,"best",best.method="youden", ret="all")
roccoord3<-coords(roc3,"best",best.method="youden", ret="all")
roccoord1
roccoord2
roccoord3

#' Sensitivity is the true positive rate and specificity is the true negative rate. 
#' The three have similar AUCs but differences come in when we look at the sensitivity 
#' and specificity. Models 1 and 3 have the highest accuracy - higher than model 2, 
#' but model 1 is the most balanced in terms of specificity. Model 3 has a higher 
#' rate of true positives but model 1 is a lot more balanced between the two. 
#' 
#' Finally we can look at the precision recall curve which is better for unbalanced 
#' data. We'll use the ROCR package. Being able to see the results of the performance 
#' object is a little weird and I had to do some digging, so if you end up doing the 
#' same digging, here is a webpage to help 
#' https://financetrain.com/measure-model-performance-in-r-using-rocr-package/ 
## ----------------------------------------------------------------------------------------------
# to use ROCR, you always start with a prediction object
# use predictions from above: pp1,pp2,pp3
# the first argument is asking for the probabilities of being positive
# so we use only the positive column from the pp1 object - in this case "end"
# the second is asking for the true labels
rocr.pred1<-prediction(pp1$end,testdf$end)
rocr.pred2<-prediction(pp2$end,testdf$end)
rocr.pred3<-prediction(pp3$end,testdf$end)

# Recall-Precision curve  
rp1<-performance(rocr.pred1,"prec","rec")
rp2<-performance(rocr.pred2,"prec","rec")
rp3<-performance(rocr.pred3,"prec","rec")

#plot
plot(rp1)
plot(rp2)
plot(rp3)

# area under 
aucpr1<-performance(rocr.pred1,"aucpr")
aucpr2<-performance(rocr.pred2,"aucpr")
aucpr3<-performance(rocr.pred3,"aucpr")
aucpr1@y.values
aucpr2@y.values
aucpr3@y.values


