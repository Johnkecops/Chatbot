pred <- function(x) {
  # Clean input text using the same preprocessing as Step 1
  corpus <- VCorpus(VectorSource(x))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)

  # Build test DTM aligned to the training vocabulary (dtm from Step 1)
  dtm_test <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(dtm)))
  data_test <- as.data.frame(as.matrix(dtm_test))

  answer <- predict(svmfit, data_test)
  paste("Answer:", as.character(answer))
}
