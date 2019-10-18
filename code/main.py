from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import wordnet
import os


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


if __name__ == "__main__":
    text_names = ["text_1", "text_2"]
    for text_name in text_names:
        if not (os.path.exists("../processed_data/"+text_name+"/lemmatizer_result.txt") and os.path.exists("../processed_data/"+text_name+"/stemmer_result.txt")):
            lemmatizer = WordNetLemmatizer()
            porter_stemmer = PorterStemmer()
            text_path = "../data/"+text_name+".txt"
            all_english_words = set(nltk.corpus.words.words())
            with open(text_path, 'r') as f:
                raw_text = f.read().lower()
                sentences = nltk.sent_tokenize(raw_text)
                words_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
                words_sentences = [[word for word in sentence if word in all_english_words] for sentence in words_sentences]
                stem_text = [[porter_stemmer.stem(word) for word in sentence] for sentence in words_sentences]
                tags_sentences = nltk.pos_tag_sents(words_sentences)
                lemmatize_text = [[lemmatizer.lemmatize(tag[0], get_wordnet_pos(tag[1])) for tag in sentence] for sentence in tags_sentences]
                lemmatize_text = [[word for word in sentence if word in all_english_words] for sentence in lemmatize_text]
            with open("../processed_data/"+text_name+"/stemmer_result.txt", 'w') as f:
                for sentence in stem_text:
                    f.write(' '.join(sentence) + '\n')
            with open("../processed_data/"+text_name+"/lemmatizer_result.txt", 'w') as f:
                for sentence in lemmatize_text:
                    f.write(' '.join(sentence) + '\n')
        else:
            stem_text = []
            with open("../processed_data/"+text_name+"/stemmer_result.txt", 'r') as f:
                for line in f.readlines():
                    sentence = line.split()
                    stem_text.append(sentence)
            lemmatize_text = []
            with open("../processed_data/"+text_name+"/lemmatizer_result.txt", 'r') as f:
                for line in f.readlines():
                    sentence = line.split()
                    lemmatize_text.append(sentence)
        stem_word_set = sorted(list(set([word for sentence in stem_text for word in sentence])))
        lemmatize_word_set = sorted(list(set([word for sentence in lemmatize_text for word in sentence])))
        with open("../dict/"+text_name+"/stemmer_word_set.txt", 'w') as f:
            f.write(str(stem_word_set))
        with open("../dict/"+text_name+"/lemmatizer_word_set.txt", 'w') as f:
            f.write(str(lemmatize_word_set))
        print(len(stem_word_set), len(lemmatize_word_set))
