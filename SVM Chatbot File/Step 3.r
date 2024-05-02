# 3. Train SVM model with the training matrix
library("e1071")
svmfit <- svm(y ~ ., data = datafix, kernel = "linear", cost = 100, scale = FALSE)
