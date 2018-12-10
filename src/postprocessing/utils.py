

def getFirstItemFromDict(dict):
    return dict[list(dict.keys())[0]]

def saveResults(results, fileName):
    path = "../doc/tables/" + fileName + ".txt"
    f = open(path, 'w')
    f.write(str(results))
    f.close()
