import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



dataset = pandas.read_csv(r"musicdata.csv", encoding="latin1")



print(dataset.shape)
print(dataset.columns)
print(dataset.isnull().sum())
print(dataset.info())

print(dataset.head())

dataset.drop(columns = 'User-Rating', inplace = True)

dataset['Song_info'] = dataset['Singer/Artists'] + dataset['Genre'] + dataset['Album/Movie']
dataset.drop(columns = ['Singer/Artists', 'Genre', 'Album/Movie'], inplace = True)

TFIDFV = TfidfVectorizer()
vectorized_Song_info = TFIDFV.fit_transform(dataset['Song_info'].values.astype('U'))



# print(dataset[dataset['Song-Name'] == 'CocaCola'].index[0])


CS = cosine_similarity(vectorized_Song_info)

def recomendation(song):
    index = dataset[dataset['Song-Name'] == song].index[0]
    recom = sorted(list(enumerate(CS[index])), reverse = True, key = lambda vector: vector[1])
    for i in recom[1 : 6]:
        print(dataset.iloc[i[0]]['Song-Name'], dataset.iloc[i[0]]['Song-Name'])
        

    
recomendation("Gali Gali")