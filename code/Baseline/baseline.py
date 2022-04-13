Results = {'truePositives': 0, 'falsePositives': 0, 'trueNegatives':0, 'falseNegatives':0}

def formatDotWords(wordsFileName, formatedFileName):
    dotWords = open(wordsFileName, "r")
    dotWordsLines = dotWords.readlines()
    formatedFile = open(formatedFileName, "w") 
    for line in dotWordsLines:
        if line != '\n':
            formatedFile.write(line)
    formatedFile.close()
    dotWords.close()


def baseLineSystem(answerKeyFileName):
    answerKeyFile = open(answerKeyFileName, "r")
    answerKeyFileLines = answerKeyFile.readlines()



    baseLineSystemResponse = False
    correctAnswer = False

    for i in range(len(answerKeyFileLines) - 2):
        baseLineSystemResponse = False
        correctAnswer = False
        if "." in answerKeyFileLines[i]:
            baseLineSystemResponse = True
        else:
            baseLineSystemResponse = False
        if answerKeyFileLines[i+1] == '\n':
            correctAnswer = True
        else:
            correctAnswer = False
    
        if baseLineSystemResponse == False and correctAnswer == True:
            Results["falseNegatives"] += 1
        elif baseLineSystemResponse == True and correctAnswer == False:
            Results["falsePositives"] += 1
        elif baseLineSystemResponse == True and correctAnswer == True:
            Results["truePositives"] += 1
        elif baseLineSystemResponse == False and correctAnswer == False:
            Results["trueNegatives"] += 1

baseLineSystem('WSJ_24.words')




precision = Results['truePositives']/(Results['truePositives'] + Results['falsePositives'])
recall = Results['truePositives']/(Results['truePositives'] + Results['falseNegatives'])
fScore = (2 * precision * recall) / (precision + recall)

print(Results)
print('Precision:', precision, 'Recall:', recall, 'fScore:', fScore)