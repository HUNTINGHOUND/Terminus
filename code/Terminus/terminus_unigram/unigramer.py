import sys
import json
import os 

def handleLine(word, pos, jDict):
    if pos in jDict.keys():
        if word in jDict[pos].keys():
            jDict[pos][word] = jDict[pos][word] + 1
        else:
            jDict[pos][word] = 1
    else:
        jDict[pos] = {word: 1}

def addToUni(path, fileName, jsonDictFileName):
    file = open(path + fileName, "r")
    jDictFile = open(jsonDictFileName, "r") 
    jDict = json.load(jDictFile)
    jDictFile.close()
    lines = file.readlines()
    file.close()
    for line in lines:
        if line != "\n":
            splitted = line.split("\t")
            if len(splitted) != 2:
                continue
            word = splitted[0]
            pos = splitted[1].strip('\n')
            handleLine(word, pos, jDict)
    jDictFile = open(jsonDictFileName, "w+") 
    json.dump(jDict, jDictFile)
    jDictFile.close()

def addToUniBrown(path, fileName, jsonDictFileName):
    file = open(path + fileName, "r")
    jDictFile = open(jsonDictFileName, "r") 
    jDict = json.load(jDictFile)
    jDictFile.close()
    lines = file.readlines()
    file.close()
    for line in lines:
        if "/"  in line:
            splitted = line.split(" ")
            for i in range(len(splitted)):
                splitted[i] = splitted[i].split("/")
                if len(splitted[i]) == 2:
                    word = splitted[i][0]
                    pos = splitted[i][1].strip("\n")
                    handleLine(word, pos, jDict)
    jDictFile = open(jsonDictFileName, "w+") 
    json.dump(jDict, jDictFile)
    jDictFile.close()




#for fileName in os.listdir('WSJ_FILES'):
#     addToUni("WSJ_FILES/", fileName, "uni.json")

for fileName in os.listdir('BROWN_FILES'):
   addToUniBrown("BROWN_FILES/", fileName, "uni.json")
