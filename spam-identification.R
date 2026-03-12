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

# Load and clean
messages <- read_tsv(file = "data/SMSSpamCollection", col_names = FALSE)
colnames(messages) <- c("type", "message")
messages <- messages |> mutate(type = factor(type, levels = c('spam', 'ham')))

# Randomly choose 20% of the data set to set aside as our test set.
# Set a seed so random selection stays the same between runs.

# TODO: Why 80%/20%??
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



# Notice that in this sample, spam messages contain phone numbers but ham
# messages do not. Is that often the case?
# First, standardize phone numbers.
convert_phone_numbers <- function(s) {
  s |>
    # Handle formats 000-000-0000, 000.000.000, and 000 000 0000
    str_replace_all("(\\d{3})[-,\\.\\s](\\d{3})[-,\\.\\s](\\d{4})",
                    "\\1\\2\\3") |>
    # Handle format (000) 000-0000
    str_replace_all("\\((\\d{3})\\)\\s?(\\d{3})-(\\d{4})", "\\1\\2\\3")
}

# Test that our conversion function is working
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

train_set <- train_set |>
  mutate(message = convert_phone_numbers(message))
test_set <- test_set |>
  mutate(message = convert_phone_numbers(message))

# Look for numbers 5-11 digits: 5 is the text shorthand (e.g. text to 87121)
# and 11 is a full phone number with country code, (08002986030).
messages_with_numbers <- train_set |> filter(str_detect(message, "\\d{5,11}"))
messages_with_numbers |> ggplot(aes(x = type, y = after_stat(count), fill =
                                      type)) + geom_bar()

# Yup, overwhelmingly messages with long numbers are spam.



# Try by most popular words
# Pick an example first to see how does.
# It's 3/10, so we'll use 310
train_set[310, ] |>
  unnest_tokens(word, message) |>
  select(word)

# Looks good, so let's try all the words.
# We should filter out stop words too to avoid just getting a bunch of prepositions.
train_set_words <- train_set |>
  unnest_tokens(word, message) |>
  filter(!word %in% stop_words$word)
head(train_set_words)

# Display the most commonly used words for ham versus spam
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

# There's a lot of overlap, but we can see some differences. "free" appears more
# frequently in spam messages, as does "txt". "gt" and "lt" appear in ham more
# often. Interestingly, "ham" also shows up a lot in ham messages, which makes
# me wonder if the classification is bleeding into the message. Let's check.

# Add row ID so we can use it to identify messages
train_set <- train_set |>
  rowid_to_column()

# Re-create the words so we can use our ID
train_set_words <- train_set |>
  unnest_tokens(word, message) |>
  filter(!word %in% stop_words$word)

# Get the IDs of messages with "ham"
ham_rows <- train_set_words |>
  filter(word == "ham") |>
  select(rowid) |>
  unique()
ham_rows

# This one looks fine.
train_set[1856, ]
# THis one, on the other hand, looks suspicious.
train_set[3650, ]

# Now we can see that it's mostly from the very long message at row 3650, which
# uses "\nham\t" and "\nspam\t". It looks like this message is actually multiple
# messages, some of which are ham and some of which are spam.

# So we need to clean it and any other messages like it up, which will change
# our spam and ham contents, so we should re-run the models after doing so.

# Find the relevant messages
# TODO: Make this a function
dirty_data <- train_set |>
  filter(str_detect(message, "\n(spam|ham)\t"))
# Take the row out of train_set
# TODO: Use some better set function here?
train_set <- train_set |>
  filter(!rowid %in% dirty_data$rowid)
# Split the data by the newline column
lines <- str_split_1(dirty_data$message, '\n')
clean_data <- str_split(lines, "\t")
first_row <- clean_data[[1]]
# If the first row doesn't contain ham/spam, then its value was already parsed
# as the type for the original row.
if (!str_detect(first_row, "^(ham|spam)")) {
  # Add the first row to the data set.
  train_set <- train_set |> add_row(type = dirty_data$type, message = parse_guess(map_chr(first_row, 1)))
  # Remove it from the processing we're going to do next
  clean_data <- clean_data[-1]
}
clean_data <- clean_data |>
  transpose() |>
  map(~ parse_guess(unlist(.))) |>
  setNames(colnames(train_set)[-1]) |>
  as.data.frame() |>
  mutate(type = factor(type, levels = levels(train_set$type)))
# Add our cleaned up data back to the train_set
# Remove the ID column we added for identification
train_set <- train_set[, -1]
train_set <- rbind(train_set, clean_data)
train_set <- train_set |>
  mutate(message = convert_phone_numbers(message))

# Now that that's straightened out, let's look at the top words again
train_set_words <- train_set |>
  unnest_tokens(word, message) |>
  filter(!word %in% stop_words$word)
head(train_set_words)

# Display the most commonly used words for ham versus spam
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

# Can still observe some differences. "gt" and "lt" appear more often in ham
# messages. "txt", "free," "claim," and "mobile" are more common in spam messages.
# Some, like "call" are common in both.

# Compute the odds of each word being in ham versus spam
ham_spam_odds <- train_set_words |>
  # Filter out numbers, we already looked at those
  filter(!str_detect(word, "\\d+")) |>
  # Counts the number of items the word appears in each type of message
  count(word, type) |>
  # Splits type into 2 columns
  spread(type, n, fill = 0) |>
  # Compute odds ratio
  mutate(or = (spam + 0.5) / (sum(spam) - spam + 0.5) /
           ((ham + 0.5) / (sum(ham) - ham + 0.5))) |>
  # Order by odds
  top_n(20, or) |>
  arrange(desc(or))
ham_spam_odds

# TODO: Graph these somehow?

# Now we can see that there are words that are far more likely to be in spam
# messages than ham messages.
top_20_spam_words <- ham_spam_odds$word

# length of messages
# Can see that on average, spam messages are slightly longer than ham messages
train_set |>
  group_by(type) |>
  summarize(avg_length = mean(nchar(message)))

# Looks like there might be both a min and a max for spam
train_set |> 
  mutate(msg_length = nchar(message)) |>
  ggplot(aes(msg_length, fill=type)) + 
  geom_histogram(binwidth=10) +
  facet_grid(type ~ .)

# Total capital letters
train_set |>
  mutate(capital_letters = str_count(message, "[A-Z]")) |>
  group_by(type) |>
  summarize(avg_capitals = mean(capital_letters), sd = sd(capital_letters))

train_set |> 
  mutate(capital_letters = str_count(message, "[A-Z]")) |>
  ggplot(aes(capital_letters, fill=type)) + 
  geom_histogram(bins=25) +
  facet_grid(type ~ .)

# Can see that messages with a lot of capital letters tend to be spam, but there's
# overlap.

# Longest string of capital letters
train_set <- train_set |>
  mutate(longest_capitals = unlist(lapply(str_match_all(message, "[A-Z]+"), function(m) {
    if (length(m) == 0) {
      0
    } else {
      max(nchar(m[,1]))
    }
  }))

train_set |>
  group_by(type) |>
  summarize(mean = mean(longest_capitals), sd = sd(longest_capitals))

# Can see that spam messages are more likely to use longer strings of capitals
train_set |> 
  ggplot(aes(longest_capitals, fill=type)) + 
  geom_histogram(binwidth=1) +
  facet_grid(type ~ .)

# Model 1: Baseline
# Just guess, assuming equal likelihood messages are spam or ham
guess_y_hat <- sample(c('spam', 'ham'), nrow(test_set), replace = TRUE) |>
  factor(levels = levels(test_set$type))
# Use F_meas to calculate the F1 score. Use beta of 0.5 because it's more
# important that a user not miss a real message.
F_meas(
  guess_y_hat,
  reference = test_set$type,
  relevant = "spam",
  beta = 0.75
)

# Model 2: Based on numbers
has_number_y_hat <- ifelse(str_detect(test_set$message, "\\d{5,11}"), "spam", "ham") |>
  factor(levels = levels(test_set$type))
F_meas(
  has_number_y_hat,
  reference = test_set$type,
  relevant = "spam",
  beta = 0.75
)

# Model 3: Based on words
has_spam_words_y_hat <- sapply(test_set$message, function(m) {
  words <- unlist(tokenize_words(m))
  if (any(words %in% top_20_spam_words)) {
    "spam"
  } else {
    "ham"
  }
}) |>
  factor(levels = levels(test_set$type))
F_meas(
  has_spam_words_y_hat,
  reference = test_set$type,
  relevant = "spam",
  beta = 0.75
)

# Model 4: Based on message length
msg_length_y_hat <- ifelse(nchar(test_set$message) < 110 | nchar(test_set$message) > 150, "ham", "spam") |>
  factor(levels = levels(test_set$type))
F_meas(
  msg_length_y_hat,
  reference = test_set$type,
  relevant = "spam",
  beta = 0.75
)

# Model 5: Based on total capital letters, using roughly the average + 1 SD
total_capitals_y_hat <- ifelse(str_count(test_set$message, "[A-Z]") > 15, "spam", "ham") |>
  factor(levels = levels(test_set$type))
F_meas(
  total_capitals_y_hat,
  reference = test_set$type,
  relevant = "spam",
  beta = 0.75
)  

# Looking at spambase, try capital letters too and length of message
# Classifciation tree
# Random forest
# other models?
# Ensemble?