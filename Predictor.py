import numpy as np

def Predictor(model, testData):
    with open("test.txt.lower.unk.unlabeled", "r") as f:
        output = open('test.txt.lower.unk.best.txt', 'w') 
        i = 0
        positiveResults = 0

        for line in f:
            testRow = testData[i, 0:-1]
            
            if np.dot(testRow.astype(int), weightVector) <= 0:
                # <= 50
                output.write(line.replace('\n',' <=50\n'))
            else:
                # >50
                output.write(line.replace('\n',' >50\n'))
                positiveResults += 1
            i += 1
        output.close()
        
    return positiveResults
