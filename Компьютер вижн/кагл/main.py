import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

# imp = SimpleImputer(missing_values=np.nan, strategy='median')


samples_to_test = 2
vectorizer = CountVectorizer()
scaler = preprocessing.StandardScaler()
new_features = []

def preprocess_data(data, fit=True):
    titles = data.text.astype(str) + data.title.astype(str)

    words_count = {'так': 1002}

    if fit:
        for line in titles.array:
            for word in line.split():
                if word not in words_count.keys():
                    words_count[word] = 0
                words_count[word] += 1
        for k in words_count:
            if words_count[k] > 150:
                new_features.append(k)

    if fit:
        transformed_texts = vectorizer.fit_transform(titles)
    else:
        transformed_texts = vectorizer.transform(titles)

    tfidf_tokens = vectorizer.get_feature_names()
    df = pd.DataFrame(data=transformed_texts.toarray(), columns=tfidf_tokens)

    df = df.filter(items=new_features)

    for i, title in df.iterrows():
        words_count = len(titles.array[0].split())
        df.loc[i] = df.loc[i] / words_count

    if fit:
        scaler.fit(df)

    df = preprocessing.normalize(scaler.transform(df))
    return df


def generate_submission(model):
    test_data = pd.read_csv("test_without_target.csv")
    test_processed = preprocess_data(test_data, False)
    test_predicted = model.predict(test_processed)

    result_table = np.array(['Id', 'Predicted'], dtype=str)

    for idx, id in np.ndenumerate(test_data['Id']):
        pred = int(test_predicted[idx])
        result_table = np.append(result_table, [str(int(id)), str(pred)])
    result_table = result_table.reshape(16184, 2)

    result_string = ""

    for row in result_table:
        result_string += ','.join(map(str, row))
        result_string += '\n'

    text_file = open('submission.csv', 'w')
    text_file.write(result_string)
    text_file.close()


full_df = pd.read_csv("train.csv").sample(frac=1)

sources = full_df[['source']].source.astype(int)

processed = preprocess_data(full_df)

X = processed[samples_to_test:]
Y = sources[samples_to_test:]

from sklearn.svm import SVC

model = SVC()
print('train')
model.fit(X, Y)
print('trained')

# search_params(model)

expected = sources[:samples_to_test].astype(int)
predicted = model.predict(processed[:samples_to_test])

print(metrics.classification_report(expected, predicted))
print(f1_score(expected, predicted, average='macro'))

generate_submission(model)
