# Libraries
if (!require(tidyverse))
  install.packages('tidyverse')
library(tidyverse)
if (!require(caret))
  install.packages('caret')
library(caret)
if (!require(tidytext))
  install.packages('tidytext')
library(tidytext)
if (!require(tokenizers))
  install.packages('tokenizers')
library(tokenizers)
if (!require(textutils))
  install.packages('textutils')
library(textutils)
if (!require(caret))
  install.packages('caret')
library(caret)
if (!require(ggridges))
  install.packages('ggridges')
library(ggridges)
if (!require(doParallel))
  install.packages(doParallel)
library(doParallel)
if (!require(foreach))
  install.packages('foreach')
library(foreach)
if (!require(rpart))
  install.packages('rpart')
library(rpart)
if (!require(randomForest))
  install.packages('randomForest')
library(randomForest)

#############################################################################
# Load and clean data
#############################################################################

# Using read_tsv() resulted in 4773 rows.
# messages <- read_tsv(file = "data/SMSSpamCollection", col_names = FALSE, col_types = c('c', 'c'))
# colnames(messages) <- c("type", "message")
# messages <- messages |> mutate(type = factor(type, levels = c('spam', 'ham')))

# Reading the lines and splitting the strings without using read_tsv gave the
# correct number of rows.
lines <- readLines("data/SMSSpamCollection")
rows <- str_split(lines, "\t")
messages <- data.frame(parse_factor(map_chr(rows, 1)), parse_character(map_chr(rows, 2)))
colnames(messages) <- c("type", "message")

# Data exploration
# Look at first 10 items
head(messages, 10) |> tibble()

# Look at the most popular words
messages |>
  unnest_tokens(word, message, to_lower=TRUE, strip_punct=TRUE) |>
  # Filter out the stop words because they're not interesting
  filter(!word %in% stop_words$word) |>
  count(word) |>
  top_n(10, n) |>
  mutate(word = reorder(word, n)) |>
  arrange(desc(n))

# Where are those lts and gts coming from? Look at some of the messages that
# contain them. tokenize_words() is what unnest_tokens uses to split the words
# into tokens. Using that here in order to use the tokens as part of the filtering
# rather than as a separate data frame.
# See @TidyText in the report
messages |>
  filter(sapply(message, function(m) {
    words <- list_c(tokenize_words(m, lowercase = TRUE, strip_punct=TRUE))
    any(c('lt', 'gt') %in% words)
  })) |>
  pull(message) |>
  head()

# Decode HTML entities and remove any HTML tags that might still be lingering.
messages <- messages |> mutate(message=HTMLrm(HTMLdecode(message)))
# Check top words (without stop words) again. That looks better.
messages |>
  unnest_tokens(word, message, to_lower=TRUE, strip_punct=TRUE) |>
  filter(!word %in% stop_words$word) |>
  count(word) |>
  top_n(10, n) |>
  mutate(word = reorder(word, n)) |>
  arrange(desc(n))

# Next, standardize phone numbers, so they're a bit easier to identify, and so
# we can look for long strings of numbers in general.
convert_phone_numbers <- function(s) {
  s |>
    # Handle formats 000-000-0000, 000.000.000, and 000 000 0000
    str_replace_all("(\\d{3})[-,\\.\\s](\\d{3})[-,\\.\\s](\\d{4})",
                    "\\1\\2\\3") |>
    # Handle format (000) 000-0000
    str_replace_all("\\((\\d{3})\\)\\s?(\\d{3})-(\\d{4})", "\\1\\2\\3")
}

# Test that the conversion function is working
phone_test_data <- c(
  'Phone number 0123456789',
  'Phone number 012-345-6789',
  'Phone number (012)345-6789',
  'Phone number (012) 345-6789',
  'Phone number 012.345.6789',
  'Phone number 012 345 6789'
)
# Should all be identical
# via https://www.tutorialpedia.org/blog/check-whether-all-elements-of-a-list-are-in-equal-in-r/
# See @Tutorialpedia in the report
all(sapply(convert_phone_numbers(phone_test_data), function(n) {
  identical(n, phone_test_data[1])
}))

# Success! So now standardize all the phone numbers in the messages.
messages <- messages |>
  mutate(message = convert_phone_numbers(message))

##############################################################################
# Analysis and variable calculation
##############################################################################



# Randomly choose 20% of the data set to set aside as our test set.
# Set a seed so random selection stays the same between runs.
set.seed(2016)
test_index <- createDataPartition(messages$type,
                                  times = 1,
                                  p = 0.2,
                                  list = FALSE)
test_set <- messages[test_index, ]
train_set <- messages[-test_index, ]


# Look at the most commonly used words in the training set, by message type.
# Are there differences?
train_set_words <- train_set |>
  unnest_tokens(word, message, to_lower=TRUE, strip_punct=TRUE) |>
  filter(!word %in% stop_words$word)

train_set_words |>
  filter(type == 'ham') |>
  count(word) |>
  top_n(10, n) |>
  mutate(word = reorder(word, n)) |>
  arrange(desc(n))

train_set_words |>
  filter(type == 'spam') |>
  count(word) |>
  top_n(10, n) |>
  mutate(word = reorder(word, n)) |>
  arrange(desc(n))

# Calculate the odds of a word being ham versus spam, so we can identify
# which words are more likely to be in spam messages, versus words that are
# likely to be any type of message.
# Compute the odds of each word being in ham versus spam
ham_spam_odds <- train_set_words |>
  # Counts the number of items the word appears in each type of message
  count(word, type) |>
  # Splits type into 2 columns
  spread(type, n, fill = 0) |>
  # Compute odds ratio, using the 0.5 correction
  # This follows https://rafalab.dfci.harvard.edu/dsbook/text-mining.html
  # See @Irizarry1 in the report
  mutate(odds_ratio = (spam + 0.5) / (sum(spam) - spam + 0.5) /
           ((ham + 0.5) / (sum(ham) - ham + 0.5))) |>
  # Order by odds
  arrange(desc(odds_ratio))
ham_spam_odds |>
  head(15) |>
  tibble()

# How many of the 7513 words have an odds ratio greater than 1?
nrow(ham_spam_odds |> filter(odds_ratio > 1))

# What about greater than 10?
nrow(ham_spam_odds |> filter(odds_ratio > 10))

# Get the words with an odds ratio of 10 or more. That's the list of words
# that are at least 10x more likely to be in a spam message than a ham message.
top_spam_words <- ham_spam_odds |> filter(odds_ratio >= 10) |> pull('word')

# Count how many times any of those words appear in the message
count_spam_words <- function(data, word_list = top_spam_words) {
  data |>
    mutate(top_spam_words_count = sapply(message, function(m) {
      words <- list_c(tokenize_words(m, lowercase = TRUE, strip_punct=TRUE))
      sum(words %in% top_spam_words)
    }))
}

train_set <- count_spam_words(train_set)
test_set <- count_spam_words(test_set)

# Now let's see how many numbers in total spam versus ham messages have and
# how long the longest string of numbers is (now that phone numbers will show
# up as one unbroken string). Are there any differences?
add_number_columns <- function(data) {
  data |>
    mutate(message = convert_phone_numbers(message)) |>
    mutate(num_count = str_count(message, "\\d")) |>
    mutate(longest_number = unlist(lapply(str_match_all(message, "\\d+"), function(m) {
      if (length(m) == 0) {
        0
      } else {
        max(nchar(m[,1]))
      }
    })))
}
train_set <- add_number_columns(train_set)
test_set <- add_number_columns(test_set)

# Box plot comparing total number of digits
train_set |> 
  ggplot(aes(type, num_count, fill=type)) +
  xlab("Message type") +
  ylab("Number of digits in message") +
  ggtitle("Comparison of digits per message by message type") +
  geom_boxplot()

# Box plot to compare longest string of numbers by type of message
train_set |> 
  ggplot(aes(type, longest_number, fill=type)) + 
  xlab("Message type") +
  ylab("Longest digits in message") +
  ggtitle("Comparison of lengths of longest string of digits per message by message type") +
  geom_boxplot()



# Count the total number of capital letters and the longest string of at least two
# characters that starts with a capital and doesnot contain a lowercase letter 
# or digit. "URGENT!! CALL ME!" should count as 17, while "Urgent!! call ME!" 
# should count as 3. (Both counts include punctuation if it's within the string.)
add_capital_columns <- function(data) {
  data |>
    mutate(capital_count = str_count(message, "[A-Z]")) |>
    mutate(longest_capitals = unlist(lapply(str_match_all(message, "[A-Z][^a-z\\d]*[A-Z]"), function(m) {
      if (length(m) == 0) {
        0
      } else {
        max(nchar(m[,1]))
      }
    })))
}

train_set <- add_capital_columns(train_set)
test_set <- add_capital_columns(test_set)

# Box plot to compare total number of capital letters by type
# There's a lot of outliers here, so not sure box plot is the right
# visualization.
train_set |> 
  ggplot(aes(type, capital_count, fill=type)) + 
  xlab("Message type") +
  ylab("Number of capital letters (A-Z)") +
  ggtitle("Comparison of number of capital letters per message by message type") +
  geom_boxplot()

# Ridge plot to compare the total number of capital letters by type
# so we can better see the shape.
train_set |> 
  ggplot(aes(capital_count, type, fill=type)) + 
  ylab("Message type") +
  xlab("Number of capital letters (A-Z)") +
  ggtitle("Comparison of total number of capital letters per message by message type") +
  geom_density_ridges(bandwidth = 2, alpha=0.5)

# Ridge plot to compare the length of the longest string
train_set |> 
  ggplot(aes(longest_capitals, type, fill=type)) + 
  xlab("# of messages") +
  ylab("Longest string of capital letters") +
  ggtitle("Comparison of length of the longest string of capital letters per message by message type") +
  geom_density_ridges(bandwidth=2, alpha=0.5)

# Histogram plot to compare the length of the longest string.
# I prefer the ridge plot here because it's easier to see the overlap. Both
# spam and ham messages might have 0 capitals, but spam has a wider distribution
# and peaks around 5-10 characters. Using the ridge above in the final report.
train_set |> 
  ggplot(aes(longest_capitals, fill=type)) + 
  xlab("# of messages") +
  ylab("Longest string of capital letters") +
  ggtitle("Comparison of length of the longest string of capital letters per message by message type") +
  geom_histogram(binwidth=5) +
  facet_wrap(type ~ .)

# Looking only at messages where the longest capital letter string is greater
# than zero, can see that while the shapes are similar, spam messages are more
# concentrated in the 0-25 character range, while ham messages, though also
# peaking in the same spot, have much more of a tail
train_set |> 
  filter(longest_capitals > 0) |>
  ggplot(aes(longest_capitals, type, fill=type)) + 
  xlab("# of messages") +
  ylab("Longest string of capital letters") +
  ggtitle("Comparison using only messages with a capital-letter string") +
  geom_density_ridges(bandwidth=2, alpha=0.5)

# Finally, what about the total length of the message?
train_set <- train_set |> mutate(msg_length = nchar(message))
test_set <- test_set |> mutate(msg_length = nchar(message))

# Can see that the median ham message is shorter than the median spam message.
# Both types contain many outliers however, so a short message is not necessarily
# ham and vice-versa.
train_set |> 
  ggplot(aes(type, msg_length, fill=type)) + 
  xlab("Message type") +
  ylab("Total message length") +
  ggtitle("Comparison of message lengths by message type") +
  geom_boxplot()

# Model 1: Baseline

# Proportion of spam messages
p_spam <- sum(train_set$type == 'spam') / nrow(train_set)

# Just guess, assuming the same proportion of spam as in the training set
guess_y_hat <- sample(c('spam', 'ham'), nrow(test_set), replace = TRUE, prob = c(p_spam, 1 - p_spam)) |>
  factor(levels = levels(test_set$type))
# Use F_meas to calculate the F1 score. Use beta of 0.75 because it's more
# important that a user not miss a real message.
guess_f_meas <- F_meas(
  guess_y_hat,
  reference = test_set$type,
  relevant = "spam",
  beta = 0.75
)
guess_f_meas

# Model 2: Longest number

# From the box plot, 5 looks like a good cutoff, but instead of just guessing,
# use the train set to maximize the F-value.
max_digit_cutoffs <- seq(0, 15)

# Run tests in parallel. Honestly, this might be overkill given the size of the
# data set, but I'm still scarred from the movie recommendation run times.
# Plus it might be more useful when when we get into the more complex models.
cores <- min(detectCores() - 1, 15)
registerDoParallel(cores)
digit_cutoff_results <- foreach(max_digit_cutoff=max_digit_cutoffs, .packages=c("caret"), .combine=c) %dopar% {
  long_number_y_hat <- ifelse(train_set$longest_number >= max_digit_cutoff, "spam", "ham") |>
    factor(levels = levels(train_set$type))
  F_meas(
    long_number_y_hat,
    reference = train_set$type,
    relevant = "spam",
    beta = 0.75
  )
}
stopImplicitCluster()
cutoff <- max_digit_cutoffs[which.max(digit_cutoff_results)]
long_number_y_hat <- ifelse(test_set$longest_number >= cutoff, "spam", "ham") |>
  factor(levels = levels(test_set$type))
long_number_f_meas <- F_meas(
  long_number_y_hat,
  reference = test_set$type,
  relevant = "spam",
  beta = 0.75
)
long_number_f_meas

# Model 3: Using logistic regression to predict spam or ham based on longest 
# number
glm_digit_length_fit <- train_set |>
  mutate(y = as.numeric(type == "spam")) |>
  glm(y ~ longest_number, data = _, family="binomial")
# Predict the results on the test set, using the model created on the training set
# Returns the likelihood that a message is spam
glm_digit_length_p_hat <- predict(glm_digit_length_fit, newdata=test_set, type="response")
# Translate those probabilities to predictions by saying that any message with
# a greater than 50% chance of being spam is spam
glm_digit_length_y_hat <- ifelse(glm_digit_length_p_hat > 0.5, "spam", "ham") |> 
  factor(levels = levels(test_set$type))
glm_digit_length_f_meas <- F_meas(
  glm_digit_length_y_hat,
  reference = test_set$type,
  relevant = "spam",
  beta = 0.75
)
glm_digit_length_f_meas

# GLM model performs less well than having a single cutoff. While spam messages
# may have more digits, the data isn't approximately bivariate normal.

# What if we look at two variables, such as number of letters and number of
# capital letters?
train_set |>
  ggplot(aes(num_count, capital_count, color = type)) +
  xlab("Total number of digits") +
  ylab("Total number of capital letters") +
  geom_point()

# Although we can see some clear groupings between spam messages and ham messages
# the boundary is non-linear.


# Create a data frame with just the features we need, for easier model fitting.
# This way, we don't have to list them out each time. Message is removed and
# top_spam_words_count converted to an unnamed vector.
train_set_features <- train_set |>
  select(!message) |>
  mutate(top_spam_words_count = unname(top_spam_words_count))

# Model 4: KNN

# Set up common parameters for controlling the train function, which we'll
# use throughout. Define the F1 statistic, calculated using the same F_meas
# function and beta as used previously, so we can use it to determine accuracy
# when tuning
control <- trainControl(summaryFunction = function(data, lev = NULL, model = NULL) {
  f_val <- F_meas(
    data$pred,
    reference = data$obs,
    relevant = "spam",
    beta = 0.75
  )
  c(F1 = f_val)
}, allowParallel = TRUE)

# Use caret 'train' method to use bootstrap samples to choose the number of neighbors
# (k) that gives us the best result. Use bootstrap here so we're choosing
# a random sampling of the *training* set, not the test set. The test set is 
# only used to check the accuracy of the final KNN model. Default value is 25 
# bootstrapped samples for each value tried
registerDoParallel(cores)
knn_fit <- train(
  type ~ ., 
  method = "knn", 
  data = train_set_features, 
  trControl = control, 
  metric = "F1",
  tuneGrid = data.frame(k = c(3, 5, 7, 9, 11))
)
knn_fit$bestTune
knn_y_hat <- predict(knn_fit, newdata = test_set, type="raw")
knn_f_meas <- F_meas(
  knn_y_hat,
  reference = test_set$type,
  relevant = "spam",
  beta = 0.75
)
knn_f_meas

# Model 5: Classification Tree
class_tree_fit <- train(
  type ~ .,
  method = "rpart",
  data=train_set_features,
  trControl = control,
  metric = "F1",
  # Try a range of complexity parameters, from 0 to 0.1
  tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
)
class_tree_y_hat <- predict(class_tree_fit, newdata = test_set, type="raw")
class_tree_f_meas <- F_meas(
  class_tree_y_hat,
  reference = test_set$type,
  relevant = "spam",
  beta = 0.75
)
class_tree_f_meas

# Graph the classification tree so we can see what rules were actually used
plot(class_tree_fit$finalModel, margin = 0.1)
text(class_tree_fit$finalModel, cex = 0.75)

# Model 6: Random forest
random_forest_fit <- train(
  type ~ .,
  method = "rf",
  data = train_set_features,
  trControl = control,
  metric = "F1",
  # Tune the number of variables randomly assigned per split
  tuneGrid=data.frame(mtry=seq(1, 6))
)
random_forest_y_hat <- predict(random_forest_fit, newdata = test_set, type="raw")
random_forest_f_meas <- F_meas(
  random_forest_y_hat,
  reference = test_set$type,
  relevant = "spam",
  beta = 0.75
)
random_forest_f_meas
stopImplicitCluster()

# Show the relative importance of various factors in the final model
importance(random_forest_fit$finalModel)


# Use bootstrap sampling
folds = createResample(train_set$type, times = 5)
guess_cv <- foreach(fold=folds, .combine = c) %do% {
  partition <- train_set[fold,]
  y_hat <- sample(c('spam', 'ham'), nrow(partition), replace = TRUE, prob = c(p_spam, 1 - p_spam)) |>
    factor(levels = levels(partition$type))
  F_meas(
    y_hat,
    reference = partition$type,
    relevant = "spam",
    beta = 0.75
  )
}
guess_accuracy <- mean(guess_cv)

longest_number_cv <- foreach(fold=folds, .combine = c) %do% {
  partition <- train_set[fold,]
  y_hat <- ifelse(partition$longest_number >= cutoff, "spam", "ham") |>
    factor(levels = levels(partition$type))
  F_meas(
    y_hat,
    reference = partition$type,
    relevant = "spam",
    beta = 0.75
  )
}
longest_number_accuracy <- mean(longest_number_cv)

glm_digit_cv <- foreach(fold=folds, .combine = c) %do% {
  partition <- train_set[fold,]
  p_hat <- predict(glm_digit_length_fit, newdata=partition, type="response")
  y_hat <- ifelse(p_hat > 0.5, "spam", "ham") |> 
    factor(levels = levels(partition$type))
  F_meas(
    y_hat,
    reference = partition$type,
    relevant = "spam",
    beta = 0.75
  )
}
glm_digit_accuracy <- mean(glm_digit_cv)

knn_cv <- foreach(fold=folds, .combine = c) %do% {
  partition <- train_set[fold,]
  y_hat <- predict(knn_fit, newdata = partition, type="raw")
  F_meas(
    y_hat,
    reference = partition$type,
    relevant = "spam",
    beta = 0.75
  )
}
knn_accuracy <- mean(knn_cv)

class_tree_cv <- foreach(fold=folds, .combine = c) %do% {
  partition <- train_set[fold,]
  y_hat <- predict(class_tree_fit, newdata = partition, type="raw")
  F_meas(
    y_hat,
    reference = partition$type,
    relevant = "spam",
    beta = 0.75
  )
}
class_tree_accuracy <- mean(class_tree_cv)

random_forest_cv <- foreach(fold=folds, .combine = c) %do% {
  partition <- train_set[fold,]
  y_hat <- predict(random_forest_fit, newdata = partition, type="raw")
  F_meas(
    y_hat,
    reference = partition$type,
    relevant = "spam",
    beta = 0.75
  )
}
random_forest_accuracy <- mean(random_forest_cv)

tibble(
  Models=c('Guessing', 'Longest number', 'Logistic regression', 'K-Nearest Neighbor', 'Classification Tree', 'Random Forest'),
  `F-Values`=c(guess_accuracy, longest_number_accuracy, glm_digit_accuracy, knn_accuracy, class_tree_accuracy, random_forest_accuracy)
)


# long_number_y_hat and glm_digit_y_hat look at the same feature, the longest
# number of digits. Try with just longest number, which did slightly better
# in my bootstrap.

# But, this gives us an even number. What to do in the case of ties? Since
# not misclassifying ham is more important than not misclassifying spam, ties
# will go to the ham side
ensemble_vote_fit <- cbind(long_number_y_hat, class_tree_y_hat, random_forest_y_hat, knn_y_hat)
ensemble_vote_y_hat <- apply(ensemble_vote_fit, 1, function(r) {
  ifelse(sum(r == 2) >= 3, 'spam', 'ham')
}) |> factor(levels = levels(test_set$type))
ensemble_vote_f_meas <- F_meas(
  ensemble_vote_y_hat,
  reference = test_set$type,
  relevant = "spam",
  beta = 0.75
)
ensemble_vote_f_meas

# Re-generate predictions to get the probability of the item being spam
knn_p_hat <- predict(knn_fit, newdata = test_set , type = "prob")['spam']
class_tree_p_hat <- predict(class_tree_fit, newdata = test_set, type = "prob")['spam']
random_forest_p_hat <- predict(random_forest_fit, newdata = test_set, type = "prob")['spam']
ensemble_mean_fit <- cbind(knn_p_hat, class_tree_p_hat, random_forest_p_hat)
# Get the mean probability across all three models. If it's greater than 0.5
# likelihood of being spam, classify it as such
ensemble_mean_y_hat <- apply(ensemble_mean_fit, 1, function(r) {
  ifelse(mean(r) > 0.5, 'spam', 'ham')
}) |> factor(levels = levels(test_set$type))
ensemble_mean_f_meas <- F_meas(
  ensemble_mean_y_hat,
  reference = test_set$type,
  relevant = "spam",
  beta = 0.75
)
ensemble_mean_f_meas


tibble(
  Models=c('Guessing', 'Longest number', 'Logistic regression', 'K-Nearest Neighbor', 'Classification Tree', 'Random Forest', 'Ensemble Voting', 'Ensemble Mean'),
  `F-Values`=c(guess_f_meas, long_number_f_meas, glm_digit_length_f_meas, knn_f_meas, class_tree_f_meas, random_forest_f_meas, ensemble_vote_f_meas, ensemble_mean_f_meas)
)
confusionMatrix(random_forest_y_hat, test_set$type, positive = "spam")