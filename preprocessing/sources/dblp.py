import preprocessing.sources.tools as nu


# Retrieve the dataset and the labels
def retrieve():
    path = 'https://raw.githubusercontent.com/shiruipan/TriDNR/master/data/dblp/'
    return nu._retrieve(path+'docs.txt', path+'labels.txt')
