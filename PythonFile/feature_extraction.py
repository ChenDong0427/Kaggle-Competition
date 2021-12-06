import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import Normalizer
import nltk 
from nltk import word_tokenize 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('wordnet')
from nltk.corpus import stopwords 
from nltk.corpus import sentiwordnet as swn 
import string 
stop = stopwords.words("english") + list(string.punctuation)
count = 0


def process(df):
    # This is where you can do all your processing

    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)

    
    print('Calculating text score...')
    df['TextScore'] = df['Text'].apply(text_score)

   
    print('Calculating summary score...')
    df['SummaryScore'] = df['Summary'].apply(text_score)


    df['TextLength'] = df['Text'].apply(text_len)
    df['TextLength'] = df['Text'].apply(text_len)
    df['SummaryLength'] = df['Summary'].apply(text_len)
    df['SummaryLength'] = df['Summary'].apply(text_len)

    
    df['Helpfulness'] = normalize(df['Helpfulness'])
    df['HelpfulnessNumerator'] = normalize(df['HelpfulnessNumerator'])
    df['HelpfulnessDenominator'] = normalize(df['HelpfulnessDenominator'])
    df['SummaryScore'] = normalize(df['SummaryScore'])
    df['TextScore'] = normalize(df['TextScore'])
    df['SummaryLength'] = normalize(df['SummaryLength'])
    df['TextLength'] = normalize(df['TextLength'])
    df['TextLength'] = normalize(df['TextLength'])
    df['SummaryLength'] = normalize(df['SummaryLength'])

    return df

def normalize(column):
    return (column-column.min())/(column.max()-column.min())

def text_len(text):
    return len(text.split(' '))

def text_score(text):
    global count
    count += 1
    print(count)
    
    ttt = nltk.pos_tag([i for i in word_tokenize(str(text).lower()) if i not in stop])
    word_tag_fq = nltk.FreqDist(ttt)
    wordlist = word_tag_fq.most_common()

    
    key = []
    part = []
    frequency = []
    for i in range(len(wordlist)):
        key.append(wordlist[i][0][0])
        part.append(wordlist[i][0][1])
        frequency.append(wordlist[i][1])
    textdf = pd.DataFrame({'key':key,
                      'part':part,
                      'frequency':frequency},
                      columns=['key','part','frequency'])

    n = ['NN','NNP','NNPS','NNS','UH']
    v = ['VB','VBD','VBG','VBN','VBP','VBZ']
    a = ['JJ','JJR','JJS']
    r = ['RB','RBR','RBS','RP','WRB']

    for i in range(len(textdf['key'])):
        z = textdf.iloc[i,1]

        if z in n:
            textdf.iloc[i,1]='n'
        elif z in v:
            textdf.iloc[i,1]='v'
        elif z in a:
            textdf.iloc[i,1]='a'
        elif z in r:
            textdf.iloc[i,1]='r'
        else:
            textdf.iloc[i,1]=''
            
        
    score = []
    for i in range(len(textdf['key'])):
        m = list(swn.senti_synsets(textdf.iloc[i,0],textdf.iloc[i,1]))
        s = 0
        ra = 0
        if len(m) > 0:
            for j in range(len(m)):
                s += (m[j].pos_score()-m[j].neg_score())/(j+1)
                ra += 1/(j+1)
            score.append(s/ra)
        else:
            score.append(0)
    if len(score)>0:
        return sum(score)
    else:
        return 0.0


# Load the dataset
trainingSet = pd.read_csv("./data/train.csv")

# Process the DataFrame
train_processed = process(trainingSet)

# Load test set
submissionSet = pd.read_csv("./data/test.csv")

# Merge on Id so that the test set can have feature columns as well
testX= pd.merge(train_processed, submissionSet, left_on='Id', right_on='Id')
testX = testX.drop(columns=['Score_x'])
testX = testX.rename(columns={'Score_y': 'Score'})

# The training set is where the score is not null
trainX =  train_processed[train_processed['Score'].notnull()]

testX.to_csv("./data/X_test.csv", index=False)
trainX.to_csv("./data/X_train.csv", index=False)
