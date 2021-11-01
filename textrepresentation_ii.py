import nltk
#nltk.download('stopwords')

d1 = "ARC is a good guy, he is not bad"
d2 = "feet wolves cooked boys girls ,!<@!"
d3 = "ARC is not a good guy, he is bad"
c1 = [d1, d2, d3]
token_d1 = nltk.word_tokenize(d1)
print(token_d1)


#remove stop words
from nltk.corpus import stopwords
stop_words_removed = [token for token in token_d1 if not token in stopwords.words('english') if token.isalpha()]
#isalpha removes every other things like punctuations
print(token_d1)
print(stop_words_removed)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer1 = CountVectorizer(min_df=2)
vectorizer1.fit(c1)
print(vectorizer1.vocabulary_)
v1 = vectorizer1.transform(c1)
print(v1.toarray())



#PRACTICE 5

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer2 = TfidfVectorizer()
vectorizer2.fit(c1)
print(vectorizer2.vocabulary_)  #look into TD-IDF separately
v2 = vectorizer2.transform(c1)
print(v2.toarray())
c2 = ["hello world", "ARC is calling"]
v_c2 = vectorizer2.transform(c2)
print(v_c2.toarray())


#PRACTICE 6 :- bag of n grams
vectorizer3 = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
vectorizer3.fit(c1)
v3 = vectorizer3.transform(c1)
print(v3.toarray())
print(vectorizer3.vocabulary_)



#PRACTICE 7 :- POS TAG
#nltk.download('averaged_perceptron_tagger')
##POS TAG
d4 = "I drink water in parties"
d5 = "I grab a drink in parties"
token4 = nltk.word_tokenize(d4)
POS_token4 = nltk.pos_tag(token4)
c2 = [d4, d5]
POS_c2 = []
for doc in c2:
    token_doc = nltk.word_tokenize(doc)
    POS_token_doc = nltk.pos_tag(token_doc)
    POS_token_temp = []
    for i in POS_token_doc:
        POS_token_temp.append(i[0] + i[1])
    POS_c2.append(" ".join(POS_token_temp))
vectorizer4 = TfidfVectorizer()
vectorizer4.fit(POS_c2)
print(vectorizer4.vocabulary_)
POS_v3 = vectorizer4.transform(POS_c2)
print(POS_v3.toarray())

