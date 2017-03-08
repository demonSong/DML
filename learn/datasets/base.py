import numpy as np

def load_hellen_appointment(return_X_y=False):
    filename = 'F:/dev_workspace/DML/learn/datasets/data/txt/hellen_appointment_dataTestSet.txt'
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    if return_X_y:
        return returnMat,classLabelVector


# classlabelvector,returnmat = load_hellen_appointment(True)
# print(classlabelvector,returnmat)


