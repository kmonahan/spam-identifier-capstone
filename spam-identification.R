# Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.

# https://statistics.arabpsychology.com/calculate-f1-score-in-r-including-example/
# https://www.geeksforgeeks.org/machine-learning/what-is-f-beta-score/
# https://www.tutorialpedia.org/blog/check-whether-all-elements-of-a-list-are-in-equal-in-r
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/evaluation-metrics.html#balanced-accuracy-and-f_1-score
# https://stackoverflow.com/questions/28467068/how-to-add-a-row-to-a-data-frame-in-r#44150746
# https://www.tidytextmining.com/tidytext.html
# https://docs.ropensci.org/tokenizers/
# https://ggplot2.tidyverse.org/reference/geom_histogram.html
# https://stackoverflow.com/questions/30111032/counting-the-number-of-capital-letters-in-each-row-in-r#30112340

# https://archive.ics.uci.edu/dataset/94/spambase

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
    mutate(top_50_word_count = sapply(message, function(m) {
      words <- list_c(tokenize_words(m, lowercase = TRUE, strip_punct=TRUE))
      sum(words %in% top_50_spam_words)
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
train_set |> 
  ggplot(aes(type, capital_count, fill=type)) + 
  xlab("Message type") +
  ylab("Number of capital letters (A-Z)") +
  ggtitle("Comparison of number of capital letters per message by message type") +
  geom_boxplot()

# Ridge plot to compare the total number of capital letters by type
train_set |> 
  ggplot(aes(capital_count, type, fill=type)) + 
  ylab("Message type") +
  xlab("Number of capital letters (A-Z)") +
  ggtitle("Comparison of number of capital letters per message by message type") +
  geom_density_ridges(bandwidth = 1, alpha=0.5)

# Box plot to compare longest string
train_set |> 
  ggplot(aes(type, longest_capitals, fill=type)) + 
  xlab("Message type") +
  ylab("Longest string of capital letters") +
  ggtitle("Comparison of lenght of the longest string of capital letters per message by message type") +
  geom_boxplot()

train_set |> 
  ggplot(aes(longest_capitals, type, fill=type)) + 
  ylab("Message type") +
  ylab("Longest string of capital letters") +
  ggtitle("Comparison of lenght of the longest string of capital letters per message by message type") +
  geom_density_ridges(bandwidth = 1.25, alpha=0.5)

# Finally, what about the total length of the message?
train_set <- train_set |> mutate(msg_length = nchar(message))
test_set <- test_set |> mutate(msg_length = nchar(message))

# We can see a difference here although, as with capital letters, it's not
# as pronounced as some of the others. But spam messages do seem to be more
# often longer than ham messages up to a point. Messages above 250 characters
# are uncommon, but they're ham when they do appear.
train_set |> ggplot(aes(type, msg_length, fill=type)) + geom_boxplot()

# Model 1: Baseline
# Just guess, assuming equal likelihood messages are spam or ham
guess_y_hat <- sample(c('spam', 'ham'), nrow(test_set), replace = TRUE) |>
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

# Model 2: Based on the longest number. Of the various factors,
# a long number had the most dramatic difference. If a number has more than 5
# digits, which is also the length of those text short numbers, assume it's spam.
long_number_y_hat <- ifelse(test_set$longest_number >= 5, "spam", "ham") |>
  factor(levels = levels(test_set$type))
long_number_f_meas <- F_meas(
  long_number_y_hat,
  reference = test_set$type,
  relevant = "spam",
  beta = 0.75
)
long_number_f_meas

# Model 3: Using linear regression with all factors
# TODO: Isn't there a way to do all but one column instead?
# TODO: Why linear regression?
lm_all_fit <- mutate(train_set, y = as.numeric(type == "spam")) |> 
  lm(y ~ top_50_word_count + num_count + longest_number + capital_count + longest_capitals + msg_length, data = _)
lm_all_p_hat <- predict(lm_all_fit, test_set)
lm_all_y_hat <- ifelse(lm_p_hat > 0.5, "spam", "ham") |> 
  factor(levels = levels(test_set$type))
lm_all_f_meas <- F_meas(
  lm_all_y_hat,
  reference = test_set$type,
  relevant = "spam",
  beta = 0.75
)
lm_all_f_meas

# TODO: Is there a point to having both model 3 and model 4?


# glm (and also review how that's different from lm)
# Loess (and also review loess)
# K-nearest-neighbors
# Classifciation tree
# Random forest
# Ensemble?

# Is the answer going to end up being just checking if there's a number that's
# 5 digits long?