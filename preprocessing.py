import spacy

nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    doc = nlp(text)
    processed_text = " ".join([token.lemma_ for token in doc if not token.is_stop])
    return processed_text