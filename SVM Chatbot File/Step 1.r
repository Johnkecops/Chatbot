setwd("C:/users/Ryan/rstudio data")
data <- read.csv('read.csv')

# 1. Convert training questions into document term matrix (sparse matrix with 1s and 0s)
library(snowballC)
library(NLP)

corpus <- Corpus(VectorSource(data$question))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, stripWhitespace)

dtm <- DocumentTermMatrix(corpus)

# convert to dataframe
dataset <- as.data.frame(as.matrix(dtm))
