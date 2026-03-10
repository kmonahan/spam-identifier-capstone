# Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.

# https://statistics.arabpsychology.com/calculate-f1-score-in-r-including-example/
# https://www.geeksforgeeks.org/machine-learning/what-is-f-beta-score/

# Libraries
if (!require(tidyverse)) install.packages('tidyverse')
library(tidyverse)
if (!require(caret)) install.packages('caret')
library(caret)

# Load and clean
messages <- read_tsv(file="data/SMSSpamCollection", col_names=FALSE)
colnames(messages) <- c("type", "message")
messages <- messages |> mutate(type=factor(type, levels=c('spam', 'ham')))

# Randomly choose 20% of the data set to set aside as our test set.
# Set a seed so random selection stays the same between runs.
set.seed(2016)
test_index <- createDataPartition(messages$type,
                                  times = 1,
                                  p = 0.2,
                                  list = FALSE)
test_set <- messages[test_index, ]
train_set <- messages[-test_index, ]

# Data exploration
# Look at first 10 items
head(train_set, 10) |> tibble()

# Model 1: Baseline
# Just guess, assuming equal likelihood messages are spam or ham
guess_y_hat <- sample(c('spam', 'ham'), nrow(test_set), replace = TRUE) |>
  factor(levels = levels(test_set$type))
# Use F_meas to calculate the F1 score. Use beta of 0.5 because it's more
# important that a user not miss a real message.
F_meas(guess_y_hat, reference=test_set$type, relevant="spam", beta=0.75)

# Notice that in this sample, spam messages contain phone numbers but ham
# messages do not. Is that often the case?
# Look for numbers 5-11 digits: 5 is the text shorthand (e.g. text to 87121)
# and 11 is a full phone number with country code, (0-800-298-6030).
messages_with_numbers <- train_set |> filter(str_detect(message, "\\d{5,11}"))
messages_with_numbers |> ggplot(aes(x=type, y=after_stat(count), fill=type)) + geom_bar()

# Model 2: Based on numbers
hasnumber_y_hat <- ifelse(str_detect(test_set$message, "\\d{5,11}"), "spam", "ham") |>
  factor(levels = levels(test_set$type))
F_meas(hasnumber_y_hat, reference=test_set$type, relevant="spam", beta=0.75)

# Try with otehr phone numbers
# Try by most popular words
# Combine the two
# Classifciation tree
# Random forest
# other models?
# Ensemble?