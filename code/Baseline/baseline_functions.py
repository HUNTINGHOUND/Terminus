import json
import pandas as pd
from regex import D


def appendPRFADF(df, to_append):
    df_length = len(df)
    df.loc[df_length] = to_append

#given dictionary of results, calculate precision, recall, fscore, accuracy
def calculatePRFA(resultsDict, name, print):
    precision = (resultsDict['truePositives'])/(resultsDict['truePositives'] + resultsDict['falsePositives'])
    recall = (resultsDict['truePositives'])/(resultsDict['truePositives'] + resultsDict['falseNegatives'])
    fscore = (2 * precision * recall) / (precision + recall)
    accuracy = (resultsDict['truePositives'] + resultsDict['trueNegatives'])/(resultsDict['truePositives'] + resultsDict['trueNegatives'] + resultsDict['falseNegatives'] + resultsDict['falsePositives'])

    if print:
        print(name + " ::: ", 'precision: ' + str(precision), 'recall: ' + str(recall), 'fscore: ' + str(fscore), 'accuracy: ' + str(accuracy))

    return [name, precision, recall, fscore, accuracy]

def makeDataFrame(listOfResultsDicts, Data_Frame):
    for pair in listOfResultsDicts:
        appendPRFADF(Data_Frame, calculatePRFA(pair[0], pair[1], False))
    Data_Frame.loc['mean'] = Data_Frame.mean()

#given system sentence splitting rule function and correct splitting rule function, and data, collect results (TP, FP, TN, FN) and store in results dictionary
def collectResults(linesList, ResultsDict, trueIdentifier, systemIdentifier):
    i = 0
    while i < len(linesList)-2:
        baseLineSystemResponse = systemIdentifier(i)
        correctAnswer = trueIdentifier(i)
        if baseLineSystemResponse == False and correctAnswer == True:
            ResultsDict["falseNegatives"] += 1
        elif baseLineSystemResponse == True and correctAnswer == False:
            ResultsDict["falsePositives"] += 1
        elif baseLineSystemResponse == True and correctAnswer == True:
            ResultsDict["truePositives"] += 1
        elif baseLineSystemResponse == False and correctAnswer == False:
            ResultsDict["trueNegatives"] += 1
        i += 1     

#Run baseline systems 
def baseLineSystemOneBrown(answerKeyFileName, ResultsBaselineOne):
    Brown_Results = ResultsBaselineOne['Brown']
    answerKeyFile = open(answerKeyFileName, "r")
    answerKeyFileLines = answerKeyFile.readlines()
    def trueIdentifier(i):
        if answerKeyFileLines[i + 1] == "==================================\n":
            return True
        return False
    def systemIdentifier(i):
         if "." in answerKeyFileLines[i]: 
             return True
         return False
    collectResults(answerKeyFileLines[16:], Brown_Results, trueIdentifier, systemIdentifier)


def baseLineSystemOneWSJ(answerKeyFileName, ResultsBaselineOne):
    WSJ_24_Results = ResultsBaselineOne['WSJ_24']
    answerKeyFile = open(answerKeyFileName, "r")
    answerKeyFileLines = answerKeyFile.readlines()
    def trueIdentifier(i):
        if answerKeyFileLines[i + 1] == "\n":
            return True
        return False
    def systemIdentifier(i):
        if "." in answerKeyFileLines[i]:
            return True
        return False 
    collectResults(answerKeyFileLines, WSJ_24_Results, trueIdentifier, systemIdentifier)

def baseLineSystemOneData_Sets(fileName, ResultsBaselineOne):
    file = open(fileName, "r")
    data = json.load(file)

    for docNumber in data.keys():
        thisDocsResultsDict = {'truePositives': 0, 'falsePositives': 0, 'trueNegatives':0, 'falseNegatives':0}
        sentenceEndings = []
        document = data[docNumber]
        for sentence in document['annotations']:
            sentenceEndings.append(int(sentence['end']))
        def trueIdentifier(i):
            if i in sentenceEndings:
                return True
            return False
        def systemIdentifier(i):
            if document['text'][(i-1)] == ".":
                return True
            return False
        collectResults([char for char in document['text']], thisDocsResultsDict, trueIdentifier, systemIdentifier)
        ResultsBaselineOne[fileName][docNumber] = thisDocsResultsDict
    
    resultsDataFrame = pd.DataFrame.from_dict(ResultsBaselineOne[fileName])
    results_df_sum = resultsDataFrame.sum(axis=1)
    ResultsBaselineOne[fileName]['Total'] = {}
    ResultsBaselineOne[fileName]['Total']['truePositives'] = results_df_sum[0]
    ResultsBaselineOne[fileName]['Total']['falsePositives'] = results_df_sum[1]
    ResultsBaselineOne[fileName]['Total']['trueNegatives'] = results_df_sum[2]
    ResultsBaselineOne[fileName]['Total']['falseNegatives'] = results_df_sum[3]

def baseLineSystemTwoData_Sets(fileName, ResultsBaselineTwo):
    file = open(fileName, "r")
    data = json.load(file)

    for docNumber in data.keys():
        thisDocsResultsDict = {'truePositives': 0, 'falsePositives': 0, 'trueNegatives':0, 'falseNegatives':0}
        sentenceEndings = []
        document = data[docNumber]
        for sentence in document['annotations']:
            sentenceEndings.append(int(sentence['end']))
        def trueIdentifier(i):
            if i in sentenceEndings:
                return True
            return False
        def systemIdentifier(i):
            if document['text'][(i-1)] == "." and document['text'][(i)] == " " and document['text'][(i+1)].isupper():
                return True
            return False
        collectResults([char for char in document['text']], thisDocsResultsDict, trueIdentifier, systemIdentifier)
        ResultsBaselineTwo[fileName][docNumber] = thisDocsResultsDict

    resultsDataFrame = pd.DataFrame.from_dict(ResultsBaselineTwo[fileName])
    results_df_sum = resultsDataFrame.sum(axis=1)
    ResultsBaselineTwo[fileName]['Total'] = {}
    ResultsBaselineTwo[fileName]['Total']['truePositives'] = results_df_sum[0]
    ResultsBaselineTwo[fileName]['Total']['falsePositives'] = results_df_sum[1]
    ResultsBaselineTwo[fileName]['Total']['trueNegatives'] = results_df_sum[2]
    ResultsBaselineTwo[fileName]['Total']['falseNegatives'] = results_df_sum[3]

def baseLineSystemThreeData_Sets(fileName, ResultsBaselineThree):
    file = open(fileName, "r")
    data = json.load(file)

    for docNumber in data.keys():
        thisDocsResultsDict = {'truePositives': 0, 'falsePositives': 0, 'trueNegatives':0, 'falseNegatives':0}
        sentenceEndings = []
        document = data[docNumber]
        for sentence in document['annotations']:
            sentenceEndings.append(int(sentence['end']))
        def trueIdentifier(i):
            if i in sentenceEndings:
                return True
            return False
        def systemIdentifier(i):
            if document['text'][(i-1)] == ";":
                return True
            return False
        collectResults([char for char in document['text']], thisDocsResultsDict, trueIdentifier, systemIdentifier)
        ResultsBaselineThree[fileName][docNumber] = thisDocsResultsDict

    resultsDataFrame = pd.DataFrame.from_dict(ResultsBaselineThree[fileName])
    results_df_sum = resultsDataFrame.sum(axis=1)
    ResultsBaselineThree[fileName]['Total'] = {}
    ResultsBaselineThree[fileName]['Total']['truePositives'] = results_df_sum[0]
    ResultsBaselineThree[fileName]['Total']['falsePositives'] = results_df_sum[1]
    ResultsBaselineThree[fileName]['Total']['trueNegatives'] = results_df_sum[2]
    ResultsBaselineThree[fileName]['Total']['falseNegatives'] = results_df_sum[3]