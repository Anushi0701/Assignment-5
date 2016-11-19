# Assignment 5
# Part 2
# Donald Trump Speech Evaluations

library(quanteda)
library(stm)
library(tm)
library(NLP)
library(openNLP)
library(ggplot2)
library(ggdendro)
library(cluster)
library(fpc)  

#load data from DonaldTrumpSpeech.csv

url<-"/Users/anushiarora/Desktop/Study Material/Semester 4/Business Analytics/Assignments/Assignment 5/Part 2/DonaldTrumpSpeech.csv"
precorpus<- read.csv(url, 
                     header=TRUE, stringsAsFactors=FALSE)
dim(precorpus) 
names(precorpus)  
head(precorpus)
str(precorpus)

# Creating a corpus for speech

require(quanteda)

speechcorpus<- corpus(precorpus$Full.Text,
                      docnames=transcriptcorpus$Documents)
#explore the corpus of speech

names(speechcorpus)   
summary(speechcorpus)  
head(speechcorpus)

#Generate DFM
corpus<- toLower(speechcorpus, keepAcronyms = FALSE) 
cleancorpus <- tokenize(corpus, 
                        removeNumbers=TRUE,  
                        removePunct = TRUE,
                        removeSeparators=TRUE,
                        removeTwitter=FALSE,
                        verbose=TRUE)


stop_words <- c("re", "net", "six", "room", "g", "gut", "oliv", "tripi","physic", "craft", "fair", "second",
                "may", "touch", "don", "voucher", "draw", "aren", "oh", "hello", "lo", "gotten", "glass","whose",
                "__they'v", "__so", "__it", "__for", "per", "novemb", "averag", "chao", "materi", "tool", "seven",
                "vet", "howev", "without", "lot", "wit", "line", "nov", "didn", "set", "abl", "would'v", "__we",
                "one", "year", "s", "t", "know", "also", "just", "like", "can", "need", "number", "say", "includ",
                "new", "go","now", "look", "back", "take", "thing", "even", "ask", "seen", "said", "put", "day",
                "anoth", "come", "use", "total", "happen", "place", "thank", "ve", "get", "much")
stop_words <- tolower(stop_words)

dfm<- dfm(cleancorpus, toLower = TRUE, 
               ignoredFeatures = c(stop_words, stopwords("english")),
               verbose=TRUE, 
               stem=TRUE)

# Reviewing top features

topfeatures(dfm, 100)   

#dfm with trigrams

cleancorpus1 <- tokenize(corpus, 
                        removeNumbers=TRUE,  
                        removePunct = TRUE,
                        removeSeparators=TRUE,
                        removeTwitter=FALSE, 
                        ngrams=3, verbose=TRUE)

dfm.trigram<- dfm(cleancorpus1, toLower = TRUE, 
                 ignoredFeatures = c(stop_words, stopwords("english")),
                 verbose=TRUE, 
                 stem=FALSE)
topfeatures.trigram<-topfeatures(dfm.trigram, n=50)
topfeatures.trigram

# Wordcloud for Speech

library(wordcloud)
set.seed(142)   #keeps cloud' shape fixed
dark2 <- brewer.pal(8, "Set1")   
freq<-topfeatures(dfm, n=100)


wordcloud(names(freq), 
          freq, max.words=200, 
          scale=c(3, .1), 
          colors=brewer.pal(8, "Set1"))

#Sentiment Analysis


mydict <- dictionary(list(positive = c("win", "love", "respect", "prestige", "power", "protect", "struggle",
                                       "enrich", "good", "survival", "morally", "movement", "strongly",
                                       "heaven", "highly-successful", "bright", "hope", "fix", "happy",
                                       "thrilled", "safety", "prosperity", "peace", "rise", "glorious", "great", "stable"),
                          negative = c("failed", "corrupt", "lie", "threat", "disastrous", "illegal", "dry",
                                       "robbed", "raided", "locked", "crooked", " illusion", "rigged", "deformed",
                                       "attack", "destroy", "destruction", "slander", "concerted", "vicious",
                                       "exposing", "corruption", "misrepresented", "horrible", "ISIS", "deport", 
                                       "incompetent", "worse", "sickness", "immorality", "guilty", "warned", "debt",
                                       "terrorist", "crime", "crushed", "poverty", "war", "conflict", "destroy", "defeat")))
dfm.sentiment <- dfm(speechcorpus, dictionary = mydict)
topfeatures(dfm.sentiment)
View(dfm.sentiment)


#running topics
temp<-textProcessor(documents=precorpus$Full.Text, metadata = precorpus)
names(temp)  # produces:  "documents", "vocab", "meta", "docs.removed" 
meta<-temp$meta
vocab<-temp$vocab
docs<-temp$documents
out <- prepDocuments(docs, vocab, meta)
docs<-out$documents
vocab<-out$vocab
meta <-out$meta

prevfit <-stm(docs , vocab , 
              K=3, 
              verbose=TRUE,
              data=meta, 
              max.em.its=25)

topics <-labelTopics(prevfit , topics=c(1:3))
topics   

plot.STM(prevfit, type="summary")
plot.STM(prevfit, type="perspectives", topics = c(1,3))
plot.STM(prevfit, type="perspectives", topics = c(1,2))
plot.STM(prevfit, type="perspectives", topics = c(2,3))

# to aid on assigment of labels & intepretation of topics

mod.out.corr <- topicCorr(prevfit)  #Estimates a graph of topic correlations
plot.topicCorr(mod.out.corr)

### Advanced method for Topic Modeling
#######################################


library(dplyr)
require(magrittr)
library(tm)
library(ggplot2)
library(stringr)
library(NLP)
library(openNLP)

#load .csv file with news articles
url<-"/Users/anushiarora/Desktop/Study Material/Semester 4/Business Analytics/Assignments/Assignment 5/Part 2/DonaldTrumpSpeech.csv"
precorpus<- read.csv(url, 
                     header=TRUE, stringsAsFactors=FALSE)

#passing Full Text to variable news_2015
speech<-precorpus$Full.Text


#Cleaning corpus
stop_words <- stopwords("SMART")
## additional junk words showing up in the data
stop_words <- c(stop_words, "said", "the", "also", "say", "just", "like","for", 
                "us", "re", "net", "six", "room", "g", "gut", "oliv", 
                "can", "may", "now", "year", "according", "mr")
stop_words <- tolower(stop_words)


speech <- gsub("'", "", speech) # remove apostrophes
speech <- gsub("[[:punct:]]", " ", speech)  # replace punctuation with space
speech <- gsub("[[:cntrl:]]", " ", speech)  # replace control characters with space
speech <- gsub("^[[:space:]]+", "", speech) # remove whitespace at beginning of documents
speech <- gsub("[[:space:]]+$", "", speech) # remove whitespace at end of documents
speech <- gsub("[^a-zA-Z -]", " ", speech) # allows only letters
speech <- tolower(speech)  # force to lowercase

## get rid of blank docs
speech <- speech[speech != ""]

# tokenize on space and output as a list:
doc.list <- strsplit(speech, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)


# remove terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
term.table <- term.table[names(term.table) != ""]
vocab <- names(term.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

#############
# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (1)
W <- length(vocab)  # number of terms in the vocab (1741)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [312, 288, 170, 436, 291, ...]
N <- sum(doc.length)  # total number of tokens in the data (56196)
term.frequency <- as.integer(term.table) 

# MCMC and model tuning parameters:
K <- 10
G <- 3000
alpha <- 0.02
eta <- 0.02

# Fit the model:
library(lda)
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
## display runtime
t2 - t1  

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

news_for_LDA <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)

library(LDAvis)
library(servr)

# create the JSON object to feed the visualization:
json <- createJSON(phi = news_for_LDA$phi, 
                   theta = news_for_LDA$theta, 
                   doc.length = news_for_LDA$doc.length, 
                   vocab = news_for_LDA$vocab, 
                   term.frequency = news_for_LDA$term.frequency)

serVis(json, out.dir = 'vis', open.browser = TRUE)

