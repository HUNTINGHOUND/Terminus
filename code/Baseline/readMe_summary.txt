This document is a summary of the results of three baseline sentence splitting systems. These are simple, manual systems for 
finding the end of a sentence in a large continuous text. The data sets are from the Brown and Wall Street Jounral Corpuses, and 
the SJVMK data sets (please see our citations for more on these data sets).


SYSTEM ONE: Splitting based on a '.' character. System one predicts the end of a sentence iff a '.' is detected.
        NAME  PRECISION    RECALL   F-SCORE  ACCURACY
0        WSJ   0.621608  0.902602  0.736204  0.974559
1      Brown   0.872549  0.500000  0.635714  0.931544
2      crime   0.374611  0.713405  0.491260  0.987586
3        bva   0.440434  0.848222  0.579807  0.990456
4         IP   0.363941  0.756523  0.491457  0.987962
5     SCOTUS   0.374052  0.726772  0.493903  0.989589
mean     NaN   0.507866  0.741254  0.571391  0.976949

SYSTEM ONE_S: System one using only the SJVMK data sets
        NAME  PRECISION    RECALL   F-SCORE  ACCURACY
0      crime   0.374611  0.713405  0.491260  0.987586
1        bva   0.440434  0.848222  0.579807  0.990456
2         IP   0.363941  0.756523  0.491457  0.987962
3     SCOTUS   0.374052  0.726772  0.493903  0.989589
mean     NaN   0.388260  0.761230  0.514107  0.988898


SYSTEM TWO: Splitting based on the regular expression '. [A-Z]'. System two predicts the end of a sentence iff a '. [A-Z]' is detected.
        NAME  PRECISION    RECALL   F-SCORE  ACCURACY
0      crime   0.684166  0.486764  0.568825  0.993800
1        bva   0.120285  0.032039  0.050600  0.990667
2         IP   0.681004  0.530208  0.596219  0.994478
3     SCOTUS   0.547910  0.474241  0.508420  0.993590
mean     NaN   0.508341  0.380813  0.431016  0.993134

SYSTEM THREE: Splitting based on a ';' character. System one predicts the end of a sentence iff a ';' is detected.
        NAME  PRECISION    RECALL   F-SCORE  ACCURACY
0      crime   0.020000  0.001330  0.002493  0.991062
1        bva   0.000000  0.000000       NaN  0.991131
2         IP   0.014493  0.001256  0.002311  0.991664
3     SCOTUS   0.003061  0.000447  0.000780  0.991997
mean     NaN   0.009388  0.000758  0.001861  0.991463