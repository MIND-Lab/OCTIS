import preprocessing.sources.source_tools as nu


def retrieve():
    """
    Retrieve the corpus, partition, edges and the labels

    Returns
    -------
    result : dictionary with corpus, training and test partitions,
             eges and labels of the corpus
    """
    path = 'https://raw.githubusercontent.com/shiruipan/TriDNR/master/data/M10/'
    return nu._retrieve(
        path+'docs.txt',
        path+'labels.txt',
        path+"adjedges.txt")
