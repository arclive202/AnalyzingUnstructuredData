import nltk
#nltk.download('punkt')
d1 = "ARC is a good guy, he is not bad"
d2 = "feet wolves cooked boys girls ,!<@!"
d3 = "ARC is not a good guy, he is bad"
c1 = [d1, d2, d3]
token_d1 = nltk.word_tokenize(d1)
print(token_d1)
tokenizer2 = nltk.tokenize.WhitespaceTokenizer()
Token_d12 = tokenizer2.tokenize(d1)
print(Token_d12)

##BOW Frequency
from sklearn.feature_extraction.text import CountVectorizer
vectorizer1 = CountVectorizer()
vectorizer1.fit(c1)
#you need to fit the vectorizer against the complete collection of words, and only then transform to get the count of the words
print(vectorizer1.vocabulary_)
v1 = vectorizer1.transform(c1)
print(v1.toarray())