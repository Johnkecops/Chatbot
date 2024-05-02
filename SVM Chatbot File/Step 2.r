# 2. Match the matrix of each training question with its corresponding answer to form a training matrix
data.train <- cbind(data['Response'], dataset)
attach(data.train)
class(Response)
y <- as.factor(Response)
class(y)

datafix <- cbind(y, dataset)
View(datafix)
