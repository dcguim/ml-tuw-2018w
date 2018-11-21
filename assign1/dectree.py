from sklearn import tree
from csv import reader

datapath = '../data/'
dataset = 'breast'

def decisionTreeModel(X, Y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X,Y)
    print(clf)
    return clf

def createReader(csvFile):
    newReader = reader(csvFile, delimiter=',')
    # Skip the header
    next(newReader,None)
    return newReader

def predict(mod,testcsv):
    pred = []
    with open(datapath + dataset + '/' + testcsv,
              newline='') as csvFile:
        bcreaderTest = createReader(csvFile)
        pred = mod.predict(list(bcreaderTest))
    return pred

def compare(pred, solcsv):
    comp = {}
    comp['pred']  = [0,0]
    comp['sol'] = [0,0]
    with open(datapath + dataset + '/' + solcsv,
              newline='') as csvFile:
        bcreaderSol = createReader(csvFile)
        testSol = list(bcreaderSol)
        for i in range(len(pred)):
            if pred[i] == 'B':
                comp['pred'][0] += 1
            elif pred[i] == 'M':
                comp['pred'][1] += 1
            if testSol[i][1] == 'B':
                comp['sol'][0] += 1
            elif testSol[i][1] == 'M':
                comp['sol'][1] += 1
    return comp

def main ():
    X = []
    y = []
    traincsv = 'bc-lrn.csv'
    iclass = 1;
    # open dataset and store values
    with open(datapath + dataset + '/' + traincsv,
              newline='') as csvFile:
        bcreaderTrain = createReader(csvFile)
        for row in bcreaderTrain:
            y.append(row[iclass])
            del row[iclass]
            X.append(row)
        clf = decisionTreeModel(X, y)
        pred = predict(clf,'bc-tes.csv')
        comp = compare(pred, 'bc-sol.csv')
        print(comp)

if '__main__':
    main()
