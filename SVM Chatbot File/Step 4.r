pred <- function(x){
  # propose a testing question and build the prediction function
  pred <- currentDTM(x)
  
  # clean the text
  corpus <- VCorpus(VectorSource(x))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus , removeNumbers)
  corpus <- tm_map(corpus , removePunctuation)
  corpus <- tm_map(corpus , stripWhitespace)
  
  # convert to dataframes
  dtm <- DocumentTermMatrix(corpus)
  data_test <- as.data.frame(as.matrix(dtm))
  
  # check here
  add_data <- data[1]
  add_data[data == 1] <- 0
  data_test <- cbind(data_test, add_data)

  # Predict the answer with the trained SVM model
  answer <- predict(svmfit, data_test)
  p <- as.character(answer)

  paste("Answer:", answer) 
}
