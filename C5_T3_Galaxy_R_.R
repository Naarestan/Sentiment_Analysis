##### Load Necessary Libraries
library(doParallel)
library(readxl)
library(dplyr)

library(tidyverse)
library(tidyr)

library(plotly)
library(corrplot)

library(caret)
library(e1071)
library(kknn)
library(readxl)
library(rmarkdown)

library(readr)
galaxySmall <- read_csv("~/Documents/Analytic@UTexas/Courses/Course 5/Task 3/smallmatrix_labeled_8d/galaxy_smallmatrix_labeled_9d.csv")
View(galaxySmall)

##### Set up Parallel Processing
# Find out how many cores there are on my laptop
detectCores()

# Create cluster with desired number of cores. 
cl <- makeCluster(2)

# Register cluster
registerDoParallel(cl)

# Confirm how many cores are now assigned to R and Rstudio
getDoParWorkers()

str(galaxySmall)
summary(galaxySmall)
sum(is.na(galaxySmall))


##### Feature Selection
# Select relevant columns for Galaxy
galaxy_relevant <- galaxySmall %>% 
  select(starts_with("galaxy"), starts_with("samsung"), galaxysentiment)
view(galaxy_relevant)

library(corrplot)
G <- cor(galaxy_relevant)
head(round(G, 2))
corrplot(G, method = "circle")
corrplot(G, method = "pie")
corrplot(G, method = "color")
corrplot(G, method = "number")

cor(galaxy_relevant$samsungcampos, galaxy_relevant$samsungcamunc) 

# Remove columns due to collinearity 
galaxy_nocoll <- galaxy_relevant %>% 
  select(-c(samsungdisunc, samsungperunc))
view(galaxy_nocoll)

# Corrplot after removal attributes due to collinearity
GG <- cor(galaxy_nocoll)
head(round(GG, 2))
corrplot(GG, method = "pie")
corrplot(GG, method = "number")

# Examine feature variance: nearZeroVar() with saveMetrics = TRUE returns an object containing a table including: frequency ratio, percentage unique, zero variance and near zero variance 
nzvMetrics_galaxy <- nearZeroVar(galaxy_nocoll, saveMetrics = TRUE)
nzvMetrics_galaxy

# nearZeroVar() with saveMetrics = FALSE returns an vector 
nzv_g <- nearZeroVar(galaxy_nocoll, saveMetrics = FALSE) 
nzv_g

# create a new data set and remove near zero variance features
galaxyNZV <- galaxy_nocoll[,-nzv_g]
str(galaxyNZV)
view(galaxyNZV)

##### Recursive Feature Elimination
# Let's sample the data before using RFE
set.seed(123)
galaxySample <- galaxy_nocoll[sample(1:nrow(galaxy_nocoll), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl_G <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

library(caret)
# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults_G <- rfe(galaxySample[,1:9], 
                    galaxySample$galaxysentiment, 
                  sizes=(1:9), 
                  rfeControl=ctrl_G)

# Get results
rfeResults_G

# Plot results
plot(rfeResults_G, type=c("g", "o"))


### Recoding Method
galaxyRC <- galaxySmall

# recode sentiment to combine factor levels 0 & 1 and 4 & 5
galaxyRC$galaxysentiment <- recode(galaxyRC$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 

# make Galaxy Sentiment a factor
galaxyRC$galaxysentiment <- as.factor(galaxyRC$galaxysentiment)

# create training and testing datasets
inT_galaxyRC <- createDataPartition(galaxyRC$galaxysentiment, 
                                    p = .70, list = FALSE)
galaxyRC_train <- galaxyRC[inT_galaxyRC, ]
galaxyRC_test <- galaxyRC[-inT_galaxyRC, ]


# apply C5.0 to train the model
galaxyC5.0_RC <- train(galaxysentiment~., data = galaxyRC_train, method = "C5.0", 
                       trControl=fitControl)

galaxyC5.0_RCpred <- predict(galaxyC5.0_RC, galaxyRC_test)

postResample(galaxyC5.0_RCpred, galaxyRC_test$galaxysentiment)


### GalaxyLargeMatrix ###
library(readr)
GalaxyLargeMatrix <- read_csv("~/Documents/Analytic@UTexas/Courses/Course 5/Task 3/GalaxyLargeMatrix.csv")
View(GalaxyLargeMatrix)

GalaxyLargeMatrix$id <- NULL
View(GalaxyLargeMatrix)

# make the prediction for iphone 
GalaxyLargeMatrix_pred <- predict(galaxyC5.0_RC, GalaxyLargeMatrix)
summary(GalaxyLargeMatrix_pred)

plot(GalaxyLargeMatrix_pred)
legend("top", title="Legend",
       c("1 - Negative","2 - Somewhat Negative","3 - Somewhat Positive", "4 - Positive"), horiz=FALSE)













# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)




