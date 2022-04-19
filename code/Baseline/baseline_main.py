from __future__ import annotations
import json
import pandas as pd
from baseline_functions import *
crime_fileName = "Baseline/data_set_SJVMK/cyber_crime.json"
bva_fileName = "Baseline/data_set_SJVMK/bva.json"
IP_fileName = "Baseline/data_set_SJVMK/intellectual_property.json"
SCOTUS_fileName = "Baseline/data_set_SJVMK/scotus.json"

'''
#Dictionaries to store results
'''

#system one
ResultsBaselineOne = {}
ResultsBaselineOne['WSJ_24'] = {'truePositives': 0, 'falsePositives': 0, 'trueNegatives':0, 'falseNegatives':0}
ResultsBaselineOne['Brown'] = {'truePositives': 0, 'falsePositives': 0, 'trueNegatives':0, 'falseNegatives':0}
ResultsBaselineOne[crime_fileName] = {}
ResultsBaselineOne[bva_fileName] = {}
ResultsBaselineOne[IP_fileName] = {}
ResultsBaselineOne[SCOTUS_fileName] = {}

#sysyem two
ResultsBaselineTwo = {}
ResultsBaselineTwo[crime_fileName] = {}
ResultsBaselineTwo[bva_fileName] = {}
ResultsBaselineTwo[IP_fileName] = {}
ResultsBaselineTwo[SCOTUS_fileName] = {}

#system three
ResultsBaselineThree = {}
ResultsBaselineThree[crime_fileName] = {}
ResultsBaselineThree[bva_fileName] = {}
ResultsBaselineThree[IP_fileName] = {}
ResultsBaselineThree[SCOTUS_fileName] = {}

'''
run Basline systems, filling dictionaries
'''
#system one
baseLineSystemOneWSJ('Baseline/WSJ_24.words', ResultsBaselineOne)
baseLineSystemOneBrown('Baseline/BROWN/tagged/cb/cb02.pos', ResultsBaselineOne)
baseLineSystemOneData_Sets(crime_fileName, ResultsBaselineOne)
baseLineSystemOneData_Sets(bva_fileName, ResultsBaselineOne)
baseLineSystemOneData_Sets(IP_fileName, ResultsBaselineOne)
baseLineSystemOneData_Sets(SCOTUS_fileName, ResultsBaselineOne)

#system two 
baseLineSystemTwoData_Sets(crime_fileName, ResultsBaselineTwo)
baseLineSystemTwoData_Sets(bva_fileName, ResultsBaselineTwo)
baseLineSystemTwoData_Sets(IP_fileName, ResultsBaselineTwo)
baseLineSystemTwoData_Sets(SCOTUS_fileName, ResultsBaselineTwo)

#system three
baseLineSystemThreeData_Sets(crime_fileName, ResultsBaselineThree)
baseLineSystemThreeData_Sets(bva_fileName, ResultsBaselineThree)
baseLineSystemThreeData_Sets(IP_fileName, ResultsBaselineThree)
baseLineSystemThreeData_Sets(SCOTUS_fileName, ResultsBaselineThree)
'''
put results in data frame
'''

#system one
BASLINESYSTEMONE = [(ResultsBaselineOne['WSJ_24'], 'WSJ'), (ResultsBaselineOne['Brown'], 'Brown'), (ResultsBaselineOne[crime_fileName]['Total'], 'crime'), (ResultsBaselineOne[bva_fileName]['Total'], 'bva'), (ResultsBaselineOne[IP_fileName]['Total'], 'IP'), (ResultsBaselineOne[SCOTUS_fileName]['Total'], 'SCOTUS')]
System_One_Data_Frame = pd.DataFrame(columns=['NAME', 'PRECISION', 'RECALL', 'F-SCORE', 'ACCURACY'])
makeDataFrame(BASLINESYSTEMONE, System_One_Data_Frame)
print("SYSTEM ONE:")
print(System_One_Data_Frame)

#system one with only data from SJVMK
BASLINESYSTEMONE_S = [(ResultsBaselineOne[crime_fileName]['Total'], 'crime'), (ResultsBaselineOne[bva_fileName]['Total'], 'bva'), (ResultsBaselineOne[IP_fileName]['Total'], 'IP'), (ResultsBaselineOne[SCOTUS_fileName]['Total'], 'SCOTUS')]
System_One_Data_Frame_S = pd.DataFrame(columns=['NAME', 'PRECISION', 'RECALL', 'F-SCORE', 'ACCURACY'])
makeDataFrame(BASLINESYSTEMONE_S, System_One_Data_Frame_S)
print("SYSTEM ONE_S:")
print(System_One_Data_Frame_S)

#system two
BASLINESYSTEMTWO = [(ResultsBaselineTwo[crime_fileName]['Total'], 'crime'), (ResultsBaselineTwo[bva_fileName]['Total'], 'bva'), (ResultsBaselineTwo[IP_fileName]['Total'], 'IP'), (ResultsBaselineTwo[SCOTUS_fileName]['Total'], 'SCOTUS')]
System_Two_Data_Frame = pd.DataFrame(columns=['NAME', 'PRECISION', 'RECALL', 'F-SCORE', 'ACCURACY'])
makeDataFrame(BASLINESYSTEMTWO, System_Two_Data_Frame)
print("SYSTEM TWO:")
print(System_Two_Data_Frame)

#system three
BASLINESYSTEMTHREE = [(ResultsBaselineThree[crime_fileName]['Total'], 'crime'), (ResultsBaselineThree[bva_fileName]['Total'], 'bva'), (ResultsBaselineThree[IP_fileName]['Total'], 'IP'), (ResultsBaselineThree[SCOTUS_fileName]['Total'], 'SCOTUS')]
System_Three_Data_Frame = pd.DataFrame(columns=['NAME', 'PRECISION', 'RECALL', 'F-SCORE', 'ACCURACY'])
makeDataFrame(BASLINESYSTEMTHREE, System_Three_Data_Frame)
print("SYSTEM THREE:")
print(System_Three_Data_Frame)