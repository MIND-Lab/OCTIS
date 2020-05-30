import preprocessing.sources.source_tools as nu


def retrieve():
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
        "citation": """@inproceedings{DBLP:conf/ijcai/PanWZZW16,
  author    = {Shirui Pan and
               Jia Wu and
               Xingquan Zhu and
               Chengqi Zhang and
               Yang Wang},
  editor    = {Subbarao Kambhampati},
  title     = {Tri-Party Deep Network Representation},
  booktitle = {Proceedings of the Twenty-Fifth International Joint Conference on
               Artificial Intelligence, {IJCAI} 2016, New York, NY, USA, 9-15 July
               2016},
  pages     = {1895--1901},
  publisher = {{IJCAI/AAAI} Press},
  year      = {2016},
  url       = {http://www.ijcai.org/Abstract/16/271},
  timestamp = {Tue, 20 Aug 2019 16:19:21 +0200},
  biburl    = {https://dblp.org/rec/conf/ijcai/PanWZZW16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}"""
    }
    return result
