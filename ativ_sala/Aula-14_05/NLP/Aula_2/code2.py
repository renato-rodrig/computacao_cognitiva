# Import WordNetLemmatizer and Counter
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk import WordNetLemmatizer
from collections import Counter
from NLP.src.nlp_utils import get_wiki_article_lower_tokens, get_english_stop_words


lower_tokens = get_wiki_article_lower_tokens()

# Retain alphabetic words: alpha_only
alpha_only = [w for w in lower_tokens if w.isalpha()]

english_stop = get_english_stop_words()
# Remove all stop words: no_stops
no_stops = [t for t in alpha_only if t not in english_stop]

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

# Create the bag-of-words: bow
bow = Counter(lemmatized)

# Print the 10 most common tokens
print(bow.most_common(10))
