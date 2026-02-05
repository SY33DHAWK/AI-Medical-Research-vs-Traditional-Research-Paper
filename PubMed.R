library(dplyr)
library(tm)
library(textclean)
library(hunspell)
library(udpipe)
library(SnowballC)
library(text2vec)
library(wordcloud)
library(RColorBrewer)
library(ggplot2)
library(dendextend)
library(Matrix)

model_info <- udpipe_download_model(language = "english")
ud_model <- udpipe_load_model(model_info$file_model)

mai_df <- read.csv("F:/DS FINAL PROJECT/ai.csv",
                   stringsAsFactors = FALSE, header = TRUE, sep = ",")
tm_df <- read.csv("F:/DS FINAL PROJECT/trad1.csv",
                  stringsAsFactors = FALSE, header = TRUE, sep = ",")

mai_text <- mai_df$Abstract
tm_text <- tm_df$Abstract

lemmatize_text <- function(text) {
  anno <- udpipe_annotate(object = ud_model, x = text)
  anno <- as.data.frame(anno)
  paste(anno$lemma, collapse = " ")
}

spell_correct <- function(text) {
  words <- unlist(strsplit(text, "\\s+"))
  corrected <- sapply(words, function(w) {
    if (hunspell_check(w)) w else {
      sug <- hunspell_suggest(w)[[1]]
      if (length(sug) > 0) sug[1] else w
    }
  })
  paste(corrected, collapse = " ")
}

clean_text <- function(text) {
  corpus <- VCorpus(VectorSource(text))
  corpus <- tm_map(corpus, content_transformer(function(x) iconv(x, "", "UTF-8", sub="")))
  corpus <- tm_map(corpus, content_transformer(replace_contraction))
  corpus <- tm_map(corpus, content_transformer(replace_emoji))
  corpus <- tm_map(corpus, content_transformer(replace_emoticon))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  text_vec <- sapply(corpus, as.character)
  text_vec <- sapply(text_vec, spell_correct)
  text_vec <- sapply(text_vec, lemmatize_text)
  corpus <- VCorpus(VectorSource(text_vec))
  corpus <- tm_map(corpus, stripWhitespace)
  corpus
}

mai_corpus <- clean_text(mai_text)
tm_corpus  <- clean_text(tm_text)
inspect(mai_corpus[1])
inspect(tm_corpus[1])

mai_clean <- sapply(mai_corpus, as.character)
tm_clean <- sapply(tm_corpus, as.character)
mai_clean <- mai_clean[!is.na(mai_clean) & nchar(mai_clean) > 0]
tm_clean <- tm_clean[!is.na(tm_clean) & nchar(tm_clean) > 0]

all_corpus <- c(mai_clean, tm_clean)

it <- itoken(all_corpus, progressbar = FALSE)
vocab <- create_vocabulary(it)
vectorizer <- vocab_vectorizer(vocab)

tfidf <- TfIdf$new()
dtm_tfidf <- create_dtm(it, vectorizer)
dtm_tfidf <- tfidf$fit_transform(dtm_tfidf)

n_mai <- length(mai_clean)
mai_tfidf <- dtm_tfidf[1:n_mai, ]
tm_tfidf  <- dtm_tfidf[(n_mai + 1):nrow(dtm_tfidf), ]

dim(mai_tfidf)
dim(tm_tfidf)
dim(dtm_tfidf)


mai_mean <- Matrix::colMeans(mai_tfidf)
tm_mean <- Matrix::colMeans(tm_tfidf)

contrastive_score <- mai_mean - tm_mean
top_features <- names(sort(contrastive_score, decreasing = TRUE))[1:300]
mai_contrastive <- mai_tfidf[, top_features]

set.seed(42)
k <- 3
kmeans_model <- kmeans(mai_contrastive, centers = k)

dist_matrix <- dist(mai_contrastive)
hc_model <- hclust(dist_matrix, method = "ward.D2")


get_top_words <- function(dtm, clusters, n = 10) {
  result <- list()
  
  for (i in unique(clusters)) {
    cluster_docs <- dtm[clusters == i, , drop = FALSE]
    cluster_mean <- Matrix::colMeans(cluster_docs)
    top_words <- names(sort(cluster_mean, decreasing = TRUE))[1:n]
    result[[paste("Cluster", i)]] <- top_words
  }
  
  return(result)
}

top_words_kmeans <- get_top_words(mai_contrastive, kmeans_model$cluster)
top_words_kmeans



for (i in names(top_words_kmeans)) {
  wordcloud(
    words = top_words_kmeans[[i]],
    freq = seq(10, 1),
    max.words = 10,
    colors = brewer.pal(8, "Dark2")
  )
  title(i)
}


for (i in names(top_words_kmeans)) {
  df <- data.frame(
    word = top_words_kmeans[[i]],
    freq = seq(10,1)
  )
  p <- ggplot(df, aes(x = reorder(word, freq), y = freq)) +
    geom_bar(stat = "identity", fill = "skyblue") +
    coord_flip() +
    ggtitle(i) +
    xlab("Words") +
    ylab("Importance") +
    theme_minimal()
  print(p)
}


pca <- prcomp(mai_contrastive, scale. = TRUE)
pca_df <- data.frame(
  PC1 = pca$x[,1],
  PC2 = pca$x[,2],
  Cluster = factor(kmeans_model$cluster)
)
ggplot(pca_df, aes(PC1, PC2, color = Cluster)) +
  geom_point(size = 3) +
  ggtitle("PCA plot")


dend <- as.dendrogram(hc_model)
dend <- color_branches(dend, k = k)
plot(dend, main = "Dendrogram")

#comparison


top_ai_features <- names(sort(contrastive_score, decreasing = TRUE))[1:20]
top_tm_features <- names(sort(contrastive_score, decreasing = FALSE))[1:20]

cat("Top AI Features:\n")
print(top_ai_features)
cat("\nTop Traditional Features:\n")
print(top_tm_features)


plot_top_words <- function(words, scores, title){
  df <- data.frame(word = words, score = scores[words])
  ggplot(df, aes(x = reorder(word, score), y = score)) +
    geom_bar(stat = "identity", fill = "skyblue") +
    coord_flip() +
    ggtitle(title) +
    xlab("Word") +
    ylab("Contrastive Score")}

plot_top_words(top_ai_features, contrastive_score, "Top AI-Dominant Words")
plot_top_words(top_tm_features, contrastive_score, "Top Traditional-Dominant Words")

wordcloud(words = top_ai_features, freq = contrastive_score[top_ai_features],
          max.words = 20, colors = brewer.pal(8, "Dark2"))
title("AI-Dominant Words")

wordcloud(words = top_tm_features, freq = abs(contrastive_score[top_tm_features]),
          max.words = 20, colors = brewer.pal(8, "Set1"))
title("Traditional-Dominant Words")

summary_stats <- data.frame(
  Corpus = c("AI", "Traditional"),
  Avg_Words_per_Doc = c(mean(nchar(mai_clean)), mean(nchar(tm_clean))),
  Total_Unique_Words = c(ncol(mai_tfidf), ncol(tm_tfidf)),
  Mean_TFIDF = c(mean(Matrix::colMeans(mai_tfidf)), mean(Matrix::colMeans(tm_tfidf)))
)
print(summary_stats)

common_words <- intersect(top_ai_features, top_tm_features)
cat("\nCommon Top Words in Both Corpora (if any):\n")
print(common_words)


df_scores <- data.frame(score = contrastive_score)
ggplot(df_scores, aes(x = score)) +
  geom_histogram(bins = 50, fill = "skyblue", color = "white") +
  ggtitle("Distribution of Contrastive Scores (AI - Traditional)") +
  xlab("Contrastive Score") +
  ylab("Frequency")
















