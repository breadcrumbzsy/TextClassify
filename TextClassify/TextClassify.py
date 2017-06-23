# -*- coding: utf-8 -*-


class TextClassify:
    def __init__(self):
        pass
    def text_classify(self, file, bow_model, classify_model):
        feature = bow_model.trainsorm_single_file(file)
        pred = classify_model.predict(feature)
        return pred

    def text_classify2(self, file, tfidf_model, classify_model):
        feature = tfidf_model.trainsorm_single_file(file)
        pred = classify_model.predict(feature)
        return pred
        