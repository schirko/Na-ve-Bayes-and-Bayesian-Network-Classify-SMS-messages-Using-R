## Classifying SMS Messages in r 

## load libraries
library(tm) #text mining package: tm_map()
library(SnowballC) #used for stemming, wordStem(), stemDocument()
library(wordcloud) #wordcloud generator
library(e1071) #Naive Bayes
library(gmodels) #CrossTable()
library(caret) #ConfusionMatrix()
library("RColorBrewer") #color palettes

## Text mining - load the spam collection data set
temp <- tempfile() #create temp var to hold spam file
download.file("http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",temp)
msgfile <- unz(temp, "SMSSpamCollection") #unzip file
sms_raw <- read.csv2(msgfile, header= FALSE, sep= "\t", quote= "", col.names= c("type","text"), stringsAsFactors= FALSE) #or use read.csv()
unlink(temp) #delete temp file

## Data exploration
str(sms_raw) # view the raw data frame structure
sms_raw$type <- factor(sms_raw$type) #convert character vector to factor
str(sms_raw$type) # view the type structure
table(sms_raw$type) # view the table recap of the raw data

## 1.1 Text Transformation and exploration
sms_corpus <- VCorpus(VectorSource(sms_raw$text)) # text mining character vectors
print(sms_corpus) # view the vectors
inspect(sms_corpus[1:2]) #summarize options
as.character(sms_corpus[[1]])
lapply(sms_corpus[1:2], as.character)
str(sms_corpus) #explore the data

# Clean data
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower)) # Convert the text to lower case
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers) # Remove numbers
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords("en")) # Remove english common stopwords
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, c("shit", "damn", "hell", "call", "now", "get", "can", "will", "just", "come", "ltgt", "â£", "txt")) # Remove stopwords
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation) # Remove punctuations
wordStem(c("learn", "learned", "computational", "computers", "computation"))
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument) # Text stemming to get word roots
sms_corpus_clean <- tm_map(sms, stripWhitespace) # Eliminate extra white spaces
as.character(sms_corpus_clean[[1]])
lapply(sms_corpus_clean[1:2], as.character)

# view the contents of before and after cleaning for comparison
as.character(sms_corpus[1:3])
as.character(sms_corpus_clean[1:3])

## 1.2 Build a term-document matrix
sms_dtm <- TermDocumentMatrix(sms_corpus_clean)
m <- as.matrix(sms_dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 5)
dim(sms_dtm)

## 1.3 Text Analysis 
#set.seed(5574) 
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=350, random.order=FALSE, rot.per=0.25, 
          colors=brewer.pal(8, "Dark2")) #create wordcloud

barplot(d[1:10,]$text, las = 2, names.arg = d[1:10,]$text,
        col ="lightblue", main ="Most frequent words",
        ylab = "Word frequencies")

## 1.4 Training & Test set
sms_dtm_train <- sms_dtm[1:4458, ] # train data is 75%
sms_dtm_test <- sms_dtm[4459:5574, ] # test data is 25%
sms_train_labels <- sms_raw[1:4458, ]$type # save vector with labels for train
sms_test_labels <- sms_raw[4459:5574, ]$type # save vector with labels for test

# Proportion for training & test labels
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

sms_freq_words <- findFreqTerms(sms_dtm, 5)
str(sms_freq_words)

sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]

convert_counts <- function(x) {x <- ifelse(x > 0, "Yes", "No")} # convert counts to Yes/No strings
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)
sms_classifier <- naiveBayes(sms_train, sms_train_labels) #build our model on the sms_train matrix
sms_test_pred <- predict(sms_classifier, sms_test) #run predictions
CrossTable(sms_test_pred, sms_test_labels, 
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual')) #compare predictions to true values

# Create confusion matrix
confusionMatrix(data = sms_test_pred, reference = sms_test_labels, positive = "ham", dnn = c("Prediction", "Actual"))

## 1.5 Print wordcloud for spam and ham
spam <- subset(sms_raw, type == "spam")
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
ham <- subset(sms_raw, type == "ham")
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))


## 1.6 Print 5 most frequenty used terms

## FAILED ATTEMPT BELOW - DOES NOT WORK
#wordcloud(spam$text, max.words = 5, scale = c(3, 0.5))
frequencies <- DocumentTermMatrix(ham)
frequencies

# look at words that appear atleast 200 times
findFreqTerms(frequencies, lowfreq = 200)
sparseWords <- removeSparseTerms(frequencies, 0.995)
sparseWords
# organizing frequency of terms
freq <- colSums(as.matrix(sparseWords))
length(freq)
ord <- order(freq)
ord 

# find associated terms
findAssocs(sparseWords, c('call','get'), corlimit=0.10)

wf <- data.frame(word = names(freq), freq = freq)
head(wf)

## Create the word cloud for each ham and spam to understand difference
ham_cloud<- which(sms_raw$text=="spam")
spam_cloud<- which(sms_raw$text=="ham")








