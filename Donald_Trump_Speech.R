# Assignment 5
# Part 2
# Donald Trump Speech Evaluations

library(quanteda)
library(stm)
library(tm)
library(NLP)
library(openNLP)

#load data from DonaldTrumpSpeech.csv
url<-
transcriptcorpus<- read.csv("/Users/anushiarora/Desktop/Study Material/Semester 4/Business Analytics/Assignments/Assignment 5/Part 2/DonaldTrumpSpeech.csv", header=TRUE, stringsAsFactors=FALSE)
dim(transcriptcorpus) 
names(transcriptcorpus)   
head(transcriptcorpus)
str(transcriptcorpus)

# Creating a corpus for speech

require(quanteda)

speechcorpus<- corpus(transcriptcorpus$Speech,
                    docnames=transcriptcorpus$Title)

#explore the corpus of speech

names(speechcorpus)   
summary(speechcorpus)  
head(speechcorpus)

#clean corpus: removes punctuation, digits, converts to lower case

speechcorpus<- toLower(speechcorpus, keepAcronyms = FALSE) 
cleancorpus <- tokenize(speechcorpus, 
                        removeNumbers=TRUE,  
                        removePunct = TRUE,
                        removeSeparators=TRUE,
                        removeTwitter=FALSE,
                        verbose=TRUE)

#explore the clean corpus

head(cleancorpus)   

#document feature matrix for clean corpus of speech

dfm.simple<- dfm(cleancorpus,
                 toLower = TRUE, 
                 ignoredFeatures = stopwords("english"), 
                 stem=TRUE,
                 verbose=FALSE)
head(dfm.simple) 

#to create a custom dictionary  list of stop words

swlist = c("re", "net", "six", "room", "g", "gut", "oliv", "tripi","physic", "craft", "fair", "second",
           "may", "touch", "don", "voucher", "draw", "aren", "oh", "hello", "lo", "gotten", "glass","whose",
           "__they'v", "__so", "__it", "__for", "per", "novemb", "averag", "chao", "materi", "tool", "seven",
           "vet", "howev", "without", "lot", "wit", "line", "nov", "didn", "set", "abl", "would'v", "__we",
           "one", "year", "s", "t", "know", "also", "just", "like", "can", "need", "number", "say", "includ",
"new", "go", "will","now", "look", "back", "take", "thing", "even", "ask", "seen", "said", "put", "day",
"anoth", "come", "use", "total", "happen", "place", "issu", "thank", "ve", "get", "much")
dfm.stem<- dfm(cleancorpus, toLower = TRUE, 
               ignoredFeatures = c(swlist, stopwords("english")),
               verbose=TRUE, 
               stem=TRUE)

#passing Full Text to variable news_2015
news_2015<-speechcorpus$Speech
news_2015 <- gsub("'", "", news_2015) # remove apostrophes
news_2015 <- gsub("[[:punct:]]", " ", news_2015)  # replace punctuation with space
news_2015 <- gsub("[[:cntrl:]]", " ", news_2015)  # replace control characters with space
news_2015 <- gsub("^[[:space:]]+", "", news_2015) # remove whitespace at beginning of documents
news_2015 <- gsub("[[:space:]]+$", "", news_2015) # remove whitespace at end of documents
news_2015 <- gsub("[^a-zA-Z -]", " ", news_2015) # allows only letters
news_2015 <- tolower(news_2015)  # force to lowercase

## get rid of blank docs
news_2015 <- news_2015[news_2015 != ""]

# tokenize on space and output as a list:
doc.list <- strsplit(news_2015, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)


# remove terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
term.table <- term.table[names(term.table) != ""]
vocab <- names(term.table)

# Wordcloud for Speech


library(wordcloud)
set.seed(142)   #keeps cloud' shape fixed
dark2 <- brewer.pal(8, "Set1")   
freq<-topfeatures(dfm.sentiment, n=100)


wordcloud(names(freq), 
          freq, max.words=200, 
          scale=c(3, .1), 
          colors=brewer.pal(8, "Set1"))


#dfm with bigrams

cleancorpus <- tokenize(transcriptcorpus, 
                        removeNumbers=TRUE,  
                        removePunct = TRUE,
                        removeSeparators=TRUE,
                        removeTwitter=FALSE, 
                        ngrams=2, verbose=TRUE)

dfm.bigram<- dfm(cleancorpus, toLower = TRUE, 
                 ignoredFeatures = c(swlist, stopwords("english")),
                 verbose=TRUE, 
                 stem=FALSE)
topfeatures.bigram<-topfeatures(dfm.bigram, n=50)
topfeatures.bigram

#Sentiment Analysis


mydict <- dictionary(list(positive = c("win", "love", "respect", "prestige", "power", "protect", "struggle",
                                       "enrich", "good", "survival", "morally", "movement", "strongly",
                                       "heaven", "highly-successful", "bright", "hope", "fix", "happy",
                                       "thrilled", "safety", "prosperity", "peace", "rise", "glorious", "great"),
                          negative = c("failed", "corrupt", "lie", "threat", "disastrous", "illegal", "dry",
                                       "robbed", "raided", "locked", "crooked", " illusion", "rigged", "deformed",
                                       "attack", "destroy", "destruction", "slander", "concerted", "vicious",
                                       "exposing", "corruption", "misrepresented", "horrible", "ISIS", "deport", 
                                       "incompetent", "worse", "sickness", "immorality", "guilty", "warned", "debt",
                                       "terrorism", "crime", "crushed", "poverty")))
dfm.sentiment <- dfm(cleancorpus, dictionary = mydict)
topfeatures(dfm.sentiment)
View(dfm.sentiment)



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
