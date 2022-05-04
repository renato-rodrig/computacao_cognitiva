# Import necessary modules
from nltk.tokenize import word_tokenize , sent_tokenize
import nltk
nltk.download('punkt')
from NLP.src.nlp_utils import get_sample_Santo_Graal

# Split scene_one into sentences: sentences
scene_one = get_sample_Santo_Graal()
sentences = sent_tokenize(scene_one,language="english")

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[3],language="english")

# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(word_tokenize(scene_one, language="english"))

# Print the unique tokens result
print(unique_tokens)