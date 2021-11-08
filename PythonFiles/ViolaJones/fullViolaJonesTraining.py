import numpy as np
import cv2
import math as mt

# To save and import images and haar features
import pickle

# Custom timer for comparison
from myTimer import Timer


#For showing
np.set_printoptions(suppress=True, precision=4)


def integralImg(img):
    if len(img.shape)==3: batched=True
    elif len(img.shape)==2: batched=False
    else: print("Image array is the wrong shape")

    if batched:
        ximg = np.cumsum(img, axis=2)
        nimg = np.cumsum(ximg, axis=1)
    else:
        ximg = np.cumsum(img, axis=1)
        nimg = np.cumsum(ximg, axis=0)

    return(nimg)

def calcRectReg(intimg, endPos, startPos=[0,0]):
    if len(intimg.shape)==3: batched=True
    elif len(intimg.shape)==2: batched=False
    else: print("Image array is the wrong shape", intimg.shape)

    endPos = np.abs(np.array(endPos))   #ensure that the end pos is positive
    if batched:
        if (startPos[0]<=0) and (startPos[1]<=0):
            return(intimg[:, endPos[1], endPos[0]])
        elif (startPos[0]<=0):
            return(intimg[:, endPos[1], endPos[0]] - 
                    intimg[:, startPos[1]-1, endPos[0]])
        elif (startPos[1]<=0):
            return(intimg[:, endPos[1], endPos[0]] - 
                    intimg[:, endPos[1], startPos[0]-1])
        else:
            return(intimg[:, endPos[1], endPos[0]] +
                    intimg[:, startPos[1]-1, startPos[0]-1] - 
                    intimg[:, startPos[1]-1, endPos[0]] - 
                    intimg[:, endPos[1], startPos[0]-1])
    else:
        if (startPos[0]==0) and (startPos[1]==0):
            return(intimg[endPos[1], endPos[0]])
        elif (startPos[0]==0):
            return(intimg[endPos[1], endPos[0]] - 
                    intimg[startPos[1]-1, endPos[0]])
        elif (startPos[1]==0):
            return(intimg[endPos[1], endPos[0]] - 
                    intimg[endPos[1], startPos[0]-1])
        else:
            return(intimg[endPos[1], endPos[0]] +
                    intimg[startPos[1]-1, startPos[0]-1] - 
                    intimg[startPos[1]-1, endPos[0]] - 
                    intimg[endPos[1], startPos[0]-1])


def getCorners(img, startX, startY, endX, endY):
    ret_vl = calcRectReg(img, [endX-1, endY-1], [startX, startY])
    if type(ret_vl)==np.ndarray:
        ret_vl = ret_vl.astype(np.float32)
    else:
        ret_vl = int(ret_vl)
    
    return(ret_vl)



def calcHaarVal(img, haar, pixelX, pixelY, haarX, haarY):
    moveX = haarX-1
    moveY = haarY-1


    if haar == 1:
        white = getCorners(img,pixelX,pixelY,pixelX+moveX,pixelY+int(mt.floor(moveY/2)))
        black = getCorners(img,pixelX,pixelY+int(mt.ceil(moveY/2)),pixelX+moveX,pixelY+moveY)
        val = white-black
    elif haar==2:
        white = getCorners(img,pixelX,pixelY,pixelX+int(mt.floor(moveX/2)),pixelY+moveY)
        black = getCorners(img,pixelX+int(mt.ceil(moveX/2)),pixelY,pixelX+moveX,pixelY+moveY)
        val = white-black
    elif haar==3:
        white1 = getCorners(img,pixelX,pixelY,pixelX+moveX,pixelY+int(mt.floor(moveY/3)))
        black = getCorners(img,pixelX,pixelY+int(mt.ceil(moveY/3)),pixelX+moveX,pixelY+int(mt.floor((moveY)*(2/3))))
        white2 = getCorners(img,pixelX,pixelY+int(mt.ceil((moveY)*(2/3))),pixelX+moveX,pixelY+moveY)
        val = white1 + white2 - black
    elif haar==4:
        white1 = getCorners(img,pixelX,pixelY,pixelX+int(mt.floor(moveX/3)),pixelY+moveY)
        black = getCorners(img,pixelX+int(mt.ceil(moveX/3)),pixelY,pixelX+int(mt.floor((moveX)*(2/3))),pixelY+moveY)
        white2 = getCorners(img,pixelX+int(mt.ceil((moveX)*(2/3))),pixelY,pixelX+moveX,pixelY+moveY)
        val = white1 + white2 - black
    elif haar==5:
        white1 = getCorners(img,pixelX,pixelY,pixelX+int(mt.floor(moveX/2)),pixelY+int(mt.floor(moveY/2)))
        black1 = getCorners(img,pixelX+int(mt.ceil(moveX/2)),pixelY,pixelX+moveX,pixelY+int(mt.floor(moveY/2)))
        black2 = getCorners(img,pixelX,pixelY+int(mt.ceil(moveY/2)),pixelX+int(mt.floor(moveX/2)),pixelY+moveY)
        white2 = getCorners(img,pixelX+int(mt.ceil(moveX/2)),pixelY+int(mt.ceil(moveY/2)),pixelX+moveX,pixelY+moveY)
        val = white1+white2-(black1+black2)
    else:
        print("haar is wrong", haar)

    return(val)

def adaboost(classifier, trainImg, imgWeights, trainYn):
    imgsSize = len(trainYn)
    faceSize = len(trainYn[trainYn==1])
    captures = np.zeros(imgsSize)
    error = 0

    haar = classifier[0]
    pixelX = classifier[1]
    pixelY = classifier[2]
    haarX = classifier[3]
    haarY = classifier[4]

    haarVals = calcHaarVal(trainImg,haar,pixelX,pixelY,haarX,haarY)

    for i in range(imgsSize):
        haarVal = haarVals[i]

        if (haarVal>= classifier[8])&(haarVal<= classifier[9]):
            if i<=faceSize-1:
                captures[i] = 1
            else:
                captures[i] = 0
                error = error + imgWeights[i]
        else:
            if i<=faceSize-1:
                captures[i] = 0
                error = error + imgWeights[i]
            else:
                captures[i] = 1

    alpha = 0.5*np.log((1-error)/error)

    for i in range(imgsSize):
        if captures[i] == 0:
            imgWeights[i] = imgWeights[i]*np.exp(alpha)
        else:
            imgWeights[i] = imgWeights[i]*np.exp(-1*alpha)

    imgWeights = imgWeights/np.sum(imgWeights)
    newWeights = imgWeights

    return(newWeights, alpha)



def loadMIT19x19():
    with open('./data/mit19x19/test.pkl', 'rb') as f:
        data = pickle.load(f)
    testImg = np.array([i[0] for i in data])
    testYn = np.array([i[1] for i in data])

    with open('./data/mit19x19/training.pkl', 'rb') as f:
        data = pickle.load(f)
    trainImg = np.array([i[0] for i in data])
    trainYn = np.array([i[1] for i in data])

    imgSize = [19,19]

    return(testImg, testYn, trainImg, trainYn, imgSize)


def ViolaJonesTraining(dataFunc, T=50):
    testImg, testYn, trainImg, trainYn, imgSize = dataFunc()

    # Calculate integral images
    MainTimer = Timer()
    MainTimer.start()
    # testImgInt = integralImg(testImg)
    trainImgInt = integralImg(trainImg)
    MainTimer.stop()

    # Initialize weights
    weights = np.ones(len(trainYn), dtype=np.float32)/len(trainYn)

    window = imgSize[0]
    haars = np.array([[1,2], [2,1], [1,3], [3,1], [2,2]])

    faceSize = len(trainYn[trainYn==1])
    facesLoc = np.where(trainYn==1)
    nonfacesLoc = np.where(trainYn==0)

    def testStop(cur, arr):
        if (cur[0]==arr[0])&(cur[1]==arr[1])&(cur[2]==arr[2])&(cur[3]==arr[3])&(cur[4]==arr[4]):
            return(True)

    newTim = Timer(5)
    newTim.start()
    for iteration in range(2):
        weakClassifier = []

        for haar in range(1,6):
            print("Working on Haar #%s"%(haar))

            dimX = haars[haar-1][0]
            dimY = haars[haar-1][1]

            newTim2 = Timer(np.arange(2,window-dimX).shape[0]*np.arange(2,window-dimY).shape[0])
            newTim2.start()
            for pixelX in range(2,window-dimX):
                for pixelY in range(2,window-dimY):
                    for haarX in np.arange(dimX, window-pixelX, dimX):
                        for haarY in np.arange(dimY, window-pixelY, dimY):
                            haarVector = calcHaarVal(trainImgInt,haar,pixelX,pixelY,haarX,haarY)
                            haarVector1 = calcHaarVal(trainImgInt[facesLoc],haar,pixelX,pixelY,haarX,haarY)
                            haarVector2 = calcHaarVal(trainImgInt[nonfacesLoc],haar,pixelX,pixelY,haarX,haarY)

                            faceMean = np.mean(haarVector1)
                            faceStd = np.std(haarVector1, ddof=1)
                            faceMax = np.max(haarVector1)
                            faceMin = np.min(haarVector1)

                            iter = np.arange(1,T+1)
                            minRating = faceMean - np.abs((iter/T)*(faceMean-faceMin))
                            maxRating = faceMean + np.abs((iter/T)*(faceMax-faceMean))
                            minRating = minRating.reshape((-1,1))
                            maxRating = maxRating.reshape((-1,1))

                            tempHaarComp = np.where((haarVector1 >= minRating)&(haarVector1 <= maxRating))
                            C = np.ones((T, weights.shape[0]))
                            C[tempHaarComp[0], tempHaarComp[1]] = 0

                            origTemp = np.vstack((tempHaarComp[0], tempHaarComp[1]))
                            

                            faceRating = np.sum(C[:,facesLoc]*weights[facesLoc], axis=2)
                            misclass = np.where(faceRating<0.05)[0].reshape((-1,1))

                            tempHaarComp = np.where((haarVector2 < minRating)|(haarVector2 > maxRating))

                            tempHaarComp = np.vstack((tempHaarComp[0], tempHaarComp[1]))
                            tempHaarComp[1] = tempHaarComp[1]+np.min(nonfacesLoc)


                            losTemp = np.where(misclass==tempHaarComp[0])
                            losTemp = np.vstack((losTemp[0], losTemp[1]))
                            tempHaarComp = tempHaarComp[:, losTemp[1]]

                            C[tempHaarComp[0], tempHaarComp[1]] = 0

                            nonFaceRating = np.sum(C[:,nonfacesLoc]*weights[nonfacesLoc], axis=2)
                            totalError = np.sum(C*weights, axis=1)

                            losTemp = sorted(np.unique(tempHaarComp[0]))
                            tempZ = np.zeros_like(totalError)

                            strongClassPos = np.where(totalError<0.5)[0]

                            if len(strongClassPos)>0:
                                strongCounter = len(strongClassPos)
                                storeRatingDiff = 1-faceRating[strongClassPos]-nonFaceRating[strongClassPos]
                                storeFaceRating = 1-faceRating[strongClassPos]
                                storeNonFaceRating = nonFaceRating[strongClassPos]
                                storeTotalError = totalError[strongClassPos]
                                storeLowerBound = minRating[strongClassPos]
                                storeUpperBound = maxRating[strongClassPos]

                                maxRatingDiff = np.max(storeRatingDiff)
                                maxRatingIndex = np.where(storeRatingDiff==maxRatingDiff)[0][0]

                                thisClassifier = [haar,pixelX,pixelY,haarX,haarY,
                                                maxRatingDiff,
                                                storeFaceRating[maxRatingIndex][0],
                                                storeNonFaceRating[maxRatingIndex][0],
                                                storeLowerBound[maxRatingIndex][0],
                                                storeUpperBound[maxRatingIndex][0],
                                                storeTotalError[maxRatingIndex]]
                                

                                weights, alpha = adaboost(thisClassifier, trainImgInt, weights.copy(), trainYn)

                                thisClassifier.append(alpha)

                                weakClassifier.append(thisClassifier)


                    newTim2.printPerTime()
                    
            print("Finished Haar #%s in %s sec"%(haar, newTim2.curr(printVal=False)))
    outfile1 = open('weights.pkl','wb')
    outfile2 = open('weak.pkl','wb')
    pickle.dump(weights,outfile1)
    pickle.dump(weakClassifier,outfile2)
    outfile1.close()
    outfile2.close()
    newTim.stop()
    exit()





ViolaJonesTraining(loadMIT19x19)


