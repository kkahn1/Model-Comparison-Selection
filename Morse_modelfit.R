#----- MODEL COMPARISON and SELECTION -----#

# Packages
library(tidyverse)
library(mlr)
library(parallelMap)
library(parallel)


#----- LOADING DATA -----#

# For this tutorial, we are using county-level data to predict whether a county voted for Trump in 2020 based on COVID-19 infection rates and demographic controls. 

# To begin, we load the data and split it into a training set and test set, indexed using a partitioning function in the package caret.

# Load data
counties = read.csv("counties.csv")

# Split dataset
set.seed(1000)
x = caret::createDataPartition(counties$trump, p=.8, list=FALSE, times=1)
train_set = counties[x,]
test_set = counties[-x,]


#----- BUILDING MODELS -----#

# Next, we use the training data to construct three models: a logistic regression, naive bayes, and support vector machine algorithm. The code for these models is not commented in detail as it has been covered in previous tutorials and in Rhys.

# Set up for models
train_set$trump = as.factor(train_set$trump) # turn into factor
task = makeClassifTask(data=train_set, target="trump") # set task

# Logistic model (see Rhys for details)
learner_logit = makeLearner("classif.logreg", predict.type="prob")
mod_logit = train(learner_logit, task)  # model object
pred_logit = predict(mod_logit, newdata=test_set)  # prediction object

# Naive Bayes model (see Rhys for details)
learner_bayes = makeLearner("classif.naiveBayes", predict.type="prob")
mod_bayes = train(learner_bayes, task)  # model object
pred_bayes = predict(mod_bayes, newdata=test_set)  # prediction object

# Support Vector Machine model (see Rhys for details)
learner_svm = makeLearner("classif.svm", predict.type="prob")
svmParamSpace = makeParamSet(
  makeDiscreteParam("kernel", values=c("polynomial", "radial", "sigmoid")),
  makeIntegerParam("degree", lower=1, upper=3),
  makeNumericParam("cost", lower=0.1, upper=10),
  makeNumericParam("gamma", lower=0.1, 10))
randSearch = makeTuneControlRandom(maxit=20)
cvForTuning = makeResampleDesc("Holdout", split=2/3)
parallelStartSocket(cpus = detectCores())
tunedSvmPars = tuneParams(makeLearner("classif.svm"), task=task,
                          resampling=cvForTuning, 
                          par.set=svmParamSpace,
                          control=randSearch)
parallelStop()
tunedSvm = setHyperPars(learner_svm, par.vals=tunedSvmPars$x)
mod_svm = train(tunedSvm, task)  # model object
pred_svm = predict(mod_svm, newdata=test_set)  # prediction object


#----- MEASURING PERFORMANCE-----#

# The performance function in mlr allows for quick calculations of model performance based on whichever measures the user inputs. See mlr-org.com for a full list of measures. Here we use accuracy (acc), area under the curve (auc), balanced accuracy (bac), balanced error rate (ber), false negative rate (fnr), false positive rate (fpr), Cohen's kappa (kappa), and mean misclassification error (mmce).

# Measure accuracy
performance(pred_logit, acc)
performance(pred_bayes, acc)
performance(pred_svm, acc)

# Other measures of performance
criteria = list(acc, auc, bac, kappa, ber, fnr, fpr, mmce)
perf_logit = performance(pred_logit, criteria)
perf_bayes = performance(pred_bayes, criteria)
perf_svm = performance(pred_svm, criteria)

# Store in a dataframe
fit = data.frame(criterion = factor(names(perf_logit), levels=names(perf_logit)), 
                 goal = c(rep("Maximize",4), rep("Minimize",4)),
                 logit = perf_logit, 
                 bayes = perf_bayes, 
                 svm = perf_svm)

# Prepare dataframe for plotting
fit2 = fit %>%
  pivot_longer(logit:svm) %>%  # reshape data
  group_by(criterion) %>%  # collapse by criterion
  mutate(best = ifelse(goal=="Maximize",  # determine which model is best fit
                       value==max(value),
                       value==min(value)))

# Plot
ggplot(fit2, aes(x=value, y=criterion, color=name)) +
  geom_point(aes(alpha=best), size=4) +
  facet_grid(goal~., scales="free") +
  labs(title="Comparing Measures of Model Performance",
       x="Value", y="Criterion", color="Model") +
  scale_color_manual(values=c("forestgreen", "tan1", "steelblue")) +
  scale_alpha_discrete(range=c(0.4, 1), guide=FALSE) +
  theme(legend.position="bottom")

# For the first four measures—ACC, AUC, BAC, and Kappa—the model with the highest value is the best fitting model. For the other four—BER, FNR, FPR, and MMCE—the model with the lowest value has the best fit. The logistic regression model has the best fit in four of these measures (AUC, BAC, Kappa, and BER), and it is tied with SVM in two others (ACC and MMCE) and with Bayes in one (FNR). The SVM has the best fit in one (FPR). With some of these measures, a tie may be a red flag of a computational error. However, the measures with ties are mathematically simple, so these ties are plausible. Overall, logistic regression seems to have the best fit, but we cannot be sure yet.


#----- VALIDATING with BOOTSTRAPPING -----#

# Another way to assess model performance is with resampling methods. Here we use bootstrapping, which is executed similarly to k-fold cross validation in Rhys. This is applied to all three models using the same measures as before.

# Define resampling strategy
boot = makeResampleDesc(method="Bootstrap", predict="both", iters=5)

# Run bootstrapping on logistic regression
boot_logit = resample(learner=learner_logit, 
                      task=task,
                      resampling=boot,
                      measures=criteria)

# Run bootstrapping on naive bayes
boot_bayes = resample(learner=learner_bayes, 
                      task=task,
                      resampling=boot,
                      measures=criteria)

# Run bootstrapping on SVM
boot_svm = resample(learner=learner_svm, 
                    task=task,
                    resampling=boot,
                    measures=criteria)

# Store in a dataframe
boot_fit = data.frame(criterion = factor(names(perf_logit), 
                                         levels=names(perf_logit)), 
                      goal = c(rep("Maximize",4), rep("Minimize",4)),
                      logit = boot_logit$aggr, 
                      bayes = boot_bayes$aggr, 
                      svm = boot_svm$aggr)

# Prepare dataframe for plotting
boot_fit2 = boot_fit %>%
  pivot_longer(logit:svm) %>%  # reshape data
  group_by(criterion) %>%  # collapse by criterion
  mutate(best = ifelse(goal=="Maximize",  # determine which model is best fit
                       value==max(value),
                       value==min(value)))

# Plot
ggplot(boot_fit2, aes(x=value, y=criterion, color=name)) +
  geom_point(aes(alpha=best), size=4) +
  facet_grid(goal~., scales="free") +
  labs(title="Comparing Measures of Model Performance (Bootstrapping)",
       x="Value", y="Criterion", color="Model") +
  scale_color_manual(values=c("forestgreen", "tan1", "steelblue")) +
  scale_alpha_discrete(range=c(0.4, 1), guide=FALSE) +
  theme(legend.position="bottom")

# Unlike before, the SVM model outperforms the other models on every measure. This provides strong evidence that SVM algorithms have the best performance for this study.

# Nathan Morse, nam@psu.edu

