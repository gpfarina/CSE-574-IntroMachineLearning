import numpy as np
import pickle as pl

def findMax():
    X=pl.load(open("resFinal.p", "rb"))
    tmpVal=X['00']
    for nH in range(4, 24, 4):
        i=0
        for lambdaval in np.linspace(0,1,10):
            lambdaval1, (w1,w2), accTrainData, accValData, accTestData=X[str(nH)+str(i)]
            lambdavalTmp, (w1Tmp,w2Tmp), accTrainDataTmp, accValDataTmp, accTestDataTmp=tmpVal
            if (accTestData> accTestDataTmp):
                tmpVal=X[str(nH)+str(i)]
                i=i+1	
    lambdaValTmp, (w1Tmp, w2Tmp), accTrainDataTmp, accValDataTmp, accTestDataTmp=tmpVal
    print(lambdaValTmp)
    Z=(lambdaValTmp, (w1Tmp, w2Tmp))
    pl.dump(Z, open("reportData.p","wb"))
    return 0

r=findMax()
