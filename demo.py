import Preprocessing.preprocessTools as ppt
import Preprocessing.Sources.newsGroup as source

dataset = source.retrieve()
minWordsInDoc = 10
minWordOccurences = 10
maxWordOccurences = 9000
preprocessed = ppt.preprocessMultiprocess(dataset,minWordsInDoc,minWordOccurences,maxWordOccurences)
ppt.save(preprocessed, "ppTest.json")