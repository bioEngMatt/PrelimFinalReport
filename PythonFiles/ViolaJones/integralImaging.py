import numpy as np

def integralImg(img):
    ximg = np.cumsum(img, axis=1)
    yimg = np.cumsum(ximg, axis=0)

    #Pad with zeros to vectorize calcs
    nimg = np.hstack((np.zeros((yimg.shape[0], 1)), yimg))
    nimg = np.vstack((np.zeros((1, nimg.shape[1])), nimg))

    return(nimg)
    
def calcRectReg(intimg, endPos, startPos=[0,0]):
    return(intimg[endPos[1]+1, endPos[0]+1] +
             intimg[startPos[1], startPos[0]] - 
             intimg[startPos[1], endPos[0]+1] - 
             intimg[endPos[1]+1, startPos[0]])