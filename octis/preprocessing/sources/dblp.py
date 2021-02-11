import octis.preprocessing.sources.source_tools as nu
import octis.configuration.citations as citations


def retrieve_dblp():
    """
    Retrieve the corpus, partition, edges and the labels

    Returns
    -------
    result : dictionary with corpus, training and test partitions,
             eges and labels of the corpus
    """
    path = 'https://raw.githubusercontent.com/shiruipan/TriDNR/master/data/dblp/'

    result = nu._retrieve(
        path+'docs.txt',
        path+'labels.txt',
        path+"adjedges.txt")

    result["info"] = {
        "name": "DBLP-Citation-network V4",
        "link": "https://github.com/shiruipan/TriDNR",
        "source": "https://github.com/shiruipan/TriDNR",
        "paper": "https://www.ijcai.org/Proceedings/16/Papers/271.pdf",
        "citation": citations.sources_dblp_M10
    }
    return result
