import preprocessing.sources.source_tools as nu


def retrieve():
    """
    Retrieve the corpus and the labels

    Returns
    -------
    result : dictionary with corpus and 
             labels of the corpus
    """
    path = 'https://raw.githubusercontent.com/shiruipan/TriDNR/master/data/M10/'
    return nu._retrieve(path+'docs.txt', path+'labels.txt')
