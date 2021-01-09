################################################################################
#                                                                              #
#               Gradient Boosting for Classification: Credit Risk              #
#                                                                              #
################################################################################
library(caret)
library(xgboost)
library(rpart)
library(partykit)
library(randomForest)
library(forecast)
library(Metrics)
library(ROCR)
library(ggplot2)
library(tictoc)
library(data.table)

setwd("...")
credit <- read.csv(file = "german_credit.csv", header = TRUE, sep = ",")

summary(credit)
head(credit)
names(credit)

set.seed(10)
# Create Train and Validationsplit into 0.6 and 0.4 partitions
inTrain <- createDataPartition(credit$Creditability, p = 0.6, list = FALSE)
train <- credit[inTrain, ]
train <- list(data = train[ ,2:ncol(train)], label = train$Creditability)

totalValidation<- credit[-inTrain, ]

# Create Validation and Testset from total Validation set so that 
# final partitions are 0.6, 0.2, 0.2
inValidation <- createDataPartition(totalValidation$Creditability, p=0.5, 
                                    list=FALSE)
# Testset
test <- totalValidation[inValidation, ]
test <- list(data = test[ ,2:ncol(test)], label = test$Creditability)

# Validationset
vali <- totalValidation[-inValidation, ]
vali <- list(data = vali[ ,2:ncol(vali)], label = vali$Creditability)

# create xgb.DMatrix objects to pass to model function
dtrain <- xgb.DMatrix(data = as.matrix(train$data), label = as.matrix(train$label))
dvali <- xgb.DMatrix(data = as.matrix(vali$data), label = as.matrix(vali$label))
dtest <- xgb.DMatrix(data = as.matrix(test$data), label = as.matrix(test$label))

scale_pos_weight <- 1 

# Create list with training and validationset
watchlist <- list(train = dtrain, vali = dvali)

# Use 5-fold-crossvalidation to find optimal number of rounds
set.seed(1)
gbRounds <- xgb.cv(data = dtrain, watchlist = watchlist,
                   eta = 0.05, nthread = 1,
                   nrounds = 500, early_stopping_rounds = 100,
                   objective = "binary:logistic", 
                   eval_metric = "error",
                   maximize = FALSE, nfold = 5,
                   scale_pos_weight = scale_pos_weight)

nRounds <- gbRounds$best_iteration
cat("Best iteration by cross validation: ", nRounds)


# Default Model Prediction
defaultModel <- xgb.train(data = dtrain, 
                          eta = 0.3, nthread = 3,
                          nrounds = nRounds, watchlist = watchlist, 
                          objective = "binary:logistic", 
                          eval_metric = "error",
                          verbose = 0, maximize = FALSE,
                          scale_pos_weigth = scale_pos_weight)

# Define plotting function for learning rates
plotLearning <- function(x){
  minTrain <- min(x$evaluation_log$train_error)
  minVali <- min(x$evaluation_log$vali_error)
  maxTrain <- max(x$evaluation_log$train_error)
  maxVali <- max(x$evaluation_log$vali_error)
  
  if (minTrain < minVali){
    lower <- minTrain
  } else {
    lower <- minVali
  }
  
  if (maxTrain > maxVali){
    upper <- maxTrain
  } else {
    upper <- maxVali
  }
  
  # Plot Learning curves
  plot(x$evaluation_log$vali_error, type = "l",
       main = "Learning Curve for Training and Validation Set", 
       xlab = "Training Round", ylab = "Prediction Error",
       ylim = c(lower, upper), col = "red")
  lines(x$evaluation_log$train_error, type = "l", col = "blue")
  legend("topright", legend=c("Validation", "Training"),
         col=c("red", "blue"), lty=1:1, cex=0.8, lwd=c(2.5,2.5))
}

# Plot learning curve for default model
plotLearning(defaultModel)

# Evaluate model on testset with accuracy
pred <- predict(defaultModel, dtest)
predDefault <- as.numeric(pred > 0.5)
confusionMatrix(as.factor(predDefault), test$Creditability)
accuracyDefault <- accuracy(test$label, predDefault)
cat("Accuracy on testset: ", accuracyDefault, "\n")


# Parameter Tuning
# create grid with hyperparameters 
maxDepth <- seq(1, 10, 1)
gamma <- seq(3, 8, 1)
subsample <- seq(0.3, 0.8, 0.1)
colsample_bytree <- seq(0.3, 0.9, 0.1)


# expand Grid for best model gridsearch
hyperGrid <- expand.grid(maxDepth = maxDepth, 
                         gamma = gamma,
                         subsample = subsample, 
                         colsample_bytree = colsample_bytree)

gradeModels <- list()

tic("XGB Tuning")
# Search Grid with loop over grid parameters and test with validation set
for (i in 1:nrow(hyperGrid)){
  # define parameters from grid
  maxDepth <- hyperGrid$maxDepth[i]
  gamma <- hyperGrid$gamma[i]
  subsample <- hyperGrid$subsample[i]
  colsample_bytree <- hyperGrid$colsample_bytree[i]

  
  # train model with parameters
  gradeModels[[i]] <- xgb.train(data = dtrain, 
                                max_depth = maxDepth, 
                                eta = 0.05, nthread = 3, gamma = gamma,
                                nrounds = nRounds, watchlist = watchlist, 
                                objective = "binary:logistic", 
                                eval_metric = "error",
                                verbose = 0, maximize = FALSE, 
                                subsample = subsample,
                                colsample_bytree = colsample_bytree,
                                scale_pos_weigth = scale_pos_weight)
}


# look for best Model by evaluating models for best score on validation set
error <- c()
for (i in 1:length(gradeModels)){
  error[i] <- gradeModels[[i]]$evaluation_log$vali_error[gbRounds$best_iteration]
}

# Save best Model
bestModel <- gradeModels[[which.min(error)]]

# Plot learning rate
plotLearning(bestModel)


# Print Model Summary
cat("------ Best Model Stats ------", "\n",
    "Max Depth: ", bestModel$params$max_depth, "\n",
    "Gamma: ", bestModel$params$gamma, "\n",
    "Subsample: ", bestModel$params$subsample, "\n",
    "Colsample by tree: ", bestModel$params$colsample_bytree, "\n",
    "Best Score: ", min(error), "\n")

# Evaluate model on testset with accuracy
pred <- predict(bestModel, dtest)
prediction <- as.numeric(pred > 0.5)

confusionMatrix(as.factor(prediction), testData$Creditability)
accuracyGB <- accuracy(testData$Creditability, prediction)
cat("Accuracy on testset: ", accuracyGB, "\n")

# Fix tuned paraameters
params <- list(max_depth = bestModel$params$max_depth,
               subsample = bestModel$params$subsample,
               colsample_bytree = bestModel$params$colsample_bytree,
               gamma = bestModel$params$gamma)

# Tune regularization
lambdaGrid <- seq(0.01, 0.1, 0.01)

regModels <- list()

for (i in 1:length(lambdaGrid)){
  lambdaReg <- lambdaGrid[i]
  print(lambdaReg)
  
  regModels[[i]] <- xgb.train(data = dtrain, 
                              params = params,
                              eta = 0.05, nthread = 3,
                              nrounds = nRounds, watchlist = watchlist, 
                              objective = "binary:logistic", 
                              eval_metric = "error",
                              verbose = 0, maximize = FALSE,
                              lambda = lambdaReg, 
                              scale_pos_weight = scale_pos_weight)
  
}

# Get errors 
error <- c()
for (i in 1:length(regModels)){
  error[i] <- regModels[[i]]$evaluation_log$vali_error[gbRounds$best_iteration]
}

# Choose best model
bestRegModel <- regModels[[which.min(error)]]
toc()

cat("optimal regularization parameter: ", bestRegModel$params$lambda)

# Plot best model
plotLearning(bestRegModel)

# Plot Learning curves for default and for tuned Model
plot(defaultModel$evaluation_log$vali_error, type = "l",
     main = "Learning Curves for default and tuned model", 
     xlab = "Training Round", ylab = "Prediction Error",
     ylim = c(min(defaultModel$evaluation_log$train_error), 
              max(defaultModel$evaluation_log$vali_error)), col = "red", lty = 2)
lines(defaultModel$evaluation_log$train_error, 
      type = "l", col = "blue", lty = 2)
lines(bestRegModel$evaluation_log$train_error, 
      type = "l", col = "blue", lwd = 2.5)
lines(bestRegModel$evaluation_log$vali_error, 
      type = "l", col = "red", lwd = 2.5)
legend(x = 45, y = 0.13, legend = c("Vali (default)", "Train (default)", 
                                   "Vali (tuned)", "Train (tuned)"),
                           col = c("red", "blue", "red",  "blue"),
       lty = c(2, 2, 1, 1), cex = 0.8, lwd = c(1, 1, 2.5, 2.5))


# Evaluate model on testset with accuracy
pred <- predict(bestRegModel, dtest)
predRegGB <- as.numeric(pred > 0.5)

confusionMatrix(as.factor(predRegGB), testData$Creditability)
accuracyRegGB <- accuracy(test$label, predRegGB)
cat("Accuracy on testset: ", accuracyRegGB, "\n")

# plot importance matrix
matGB <- xgb.importance(feature_names = colnames(dtrain), model = bestRegModel)
importanceGB <- xgb.ggplot.importance(importance_matrix = matGB[1:20], 
                                      rel_to_first = TRUE)  
importanceGB



############################## Random Forest ###################################

# Prepare Data for functions
trainData <- data.frame(train$label, train$data)
trainData$train.label <- as.factor(trainData$train.label)
names(trainData) <- names(credit)
valiData <- data.frame(vali$label, vali$data)
valiData$vali.label <- as.factor(valiData$vali.label)
names(valiData) <- names(credit)
testData <- data.frame(test$label, test$data)
testData$test.label <- as.factor(testData$test.label)
names(testData) <- names(credit)


# Establish a list of possible values for mtry, nodesize and sampsize
mtry <- seq(2, 8, 2)
nodesize <- seq(3, 8, 1)
sampsize <- nrow(trainData) * c(0.5 ,0.7, 0.8)
ntree <- seq(250, 500, 50)

# Create a data frame containing all combinations 
hyper_grid <- expand.grid(mtry = mtry, nodesize = nodesize, 
                          sampsize = sampsize, ntree = ntree)

# Create an empty vector to store OOB error values
oob_err <- c()

# Write a loop over the rows of hyper_grid to train the grid of models
tic("Random Forest Tuning")
for (i in 1:nrow(hyper_grid)){
  
  # Train a Random Forest model
  set.seed(2)
  model <- randomForest(formula = Creditability ~ ., 
                        data = trainData, 
                        ntree = hyper_grid$ntree[i],
                        mtry = hyper_grid$mtry[i],
                        nodesize = hyper_grid$nodesize[i],
                        sampsize = hyper_grid$sampsize[i])
  
  oob_err[i] <- model$err.rate[nrow(model$err.rate), "OOB"]
}

# Identify optimal set of hyperparmeters based on OOB error
opt_i <- which.min(oob_err)
bestParams <- hyper_grid[opt_i,]

set.seed(21)
bestRF <- randomForest(formula = Creditability ~ ., 
                       data = trainData,
                       mtry = bestParams$mtry,
                       nodesize = bestParams$nodesize,
                       sampsize = bestParams$sampsize, 
                       ntree = bestParams$ntree)
toc()

# Plot learning curve
plot(bestRF$err.rate[,"0"], main = "Learning Curve - Random Forrest", 
     col = "red", ylab = "Error Rate", xlab = "No. of Trees", type = "l", 
     ylim = c(min(bestRF$err.rate[,"1"]), max(bestRF$err.rate[,"0"])))
lines(bestRF$err.rate[, "OOB"], col = "black")
lines(bestRF$err.rate[, "1"], col = "green")
legend(x = 350, y = 0.45, colnames(model$err.rate),col=1:4,cex=0.8,fill=1:4)


# Evaluate Model on Test Set
predRF <- predict(bestRF, newdata = testData, type = "class")
confusionMatrix(predRF, testData$Creditability)
accuracyRF <- accuracy(testData$Creditability, predRF)

# Plot Importance matrix
featureImp <- sort(importance(bestRF)[,1])/max(importance(bestRF)[,1])
featureImp <- data.frame(x1 = labels(featureImp), y1 = featureImp)
featureImp <- transform(featureImp, x1 = reorder(x1, y1))

matRF <- as.data.table(featureImp)
colnames(matRF) <- colnames(matGB)[1:2]
importanceRF <- xgb.ggplot.importance(importance_matrix = matRF[1:20])

importanceRF

################################ Pruned Tree ###################################

set.seed(3)
tree<-rpart(Creditability~., data = trainData,
            method = "class", minsplit = 10, cp = 0)

# Prune the tree
cpBest <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
cat("---Complexity Parameters: ---", "\n",
    "best cv: ", round(cpBest, 3), "\n",
    "Xerror: ", round(tree$cptable[which.min(tree$cptable[,"xerror"]),"xerror"], 3))
bestTree<- prune(tree, cp = cpBest) 

# Plot the pruned regression tree
plot(as.party(bestTree))

# Evaluate Tree Performance on test set
predTree <- predict(bestTree, newdata = testData, type = "class")
confusionMatrix(predTree, testData$Creditability)
accuracyTree <- accuracy(testData$Creditability, predTree)

############################# Model Comparison #################################

# Display Model Accuracies
cat("- Accuracy of different models on test set - \n",
    "Pruned Tree: ", accuracyTree, "\n",
    "Random Forest: ", accuracyRF, "\n",
    "XGBoost (tuned): ", accuracyRegGB, "\n",
    "XGBoost (default): ", accuracyDefault, "\n")

# Dispaly AUC-Scores
cat("-- Area under the Curve -- ", "\n",
    paste("Decision Tree, AUC: ", 
          round(auc(predTree, testData$Creditability), 2)), "\n", 
    paste("Random Forest, AUC: ", 
          round(auc(predRF, testData$Creditability), 2)), "\n",
    paste("Boosting (tuned), AUC: ", 
          round(auc(predRegGB, testData$Creditability), 2)), "\n",
    paste("Boosting (default), AUC: ", 
          round(auc(predDefault, testData$Creditability), 2)), "\n")

