import numpy as np
import random as RD

class Affine(object):
    @classmethod
    def transPntForward(cls, pt, T):
        newPt = np.zeros(2, dtype=pt.dtype)
        newPt[0] = T[0,0]*pt[0]+T[1,0]*pt[1]+T[2,0]
        newPt[1] = T[0,1]*pt[0]+T[1,1]*pt[1]+T[2,1]
        return newPt

    @classmethod
    def transPntsForwardWithSameT(cls, pts, T):
        if pts.ndim != 2:
            raise Exception("Must 2-D array")
        newPts = np.zeros(pts.shape)
        newPts[:,0] = T[0,0]*pts[:,0]+T[1,0]*pts[:,1]+T[2,0]
        newPts[:,1] = T[0,1]*pts[:,0]+T[1,1]*pts[:,1]+T[2,1]
        return newPts

    @classmethod
    def transPntsForwardWithDiffT(cls, pts, Ts):
        if pts.ndim != 2:
            raise Exception("Must 2-D array")
        nPts = np.zeros(pts.shape)
        pntNum = pts.shape[0]

        for i in range(pntNum):
            T = Ts[i]
            nPts[i,0]=T[0,0]*pts[i,0]+T[1,0]*pts[i,1]+T[2,0]
            nPts[i,1]=T[0,1]*pts[i,0]+T[1,1]*pts[i,1]+T[2,1]
        return nPts

    @classmethod
    def fitGeoTrans(cls, src, dst, 
                    mode="NonreflectiveSimilarity"):
        """
        This function is the same as matlab fitgeotrans 
        """

        ### Subtract the mean
        src0 = np.subtract(src, np.mean(src, axis=0))
        dst0 = np.subtract(dst, np.mean(dst, axis=0))

        if "NonreflectiveSimilarity" == mode:
            return cls.findNonreflectiveSimilarity(src0, dst0)
        else:
            raise Exception("Unsupported transformation")

    @classmethod
    def findNonreflectiveSimilarity(cls, uv, xy):
        uv, normMatSrc = cls.normalizeControlPoints(uv)
        xy, normMatDst = cls.normalizeControlPoints(xy)
        ptNum = uv.shape[0]
        minNonCollinearPairs = 2

        x = xy[:, 0].reshape(ptNum, 1)
        y = xy[:, 1].reshape(ptNum, 1)
        X = np.concatenate((
                np.concatenate((x, y, 
                                np.ones((ptNum, 1)), 
                                np.zeros((ptNum, 1))),axis=1),
                np.concatenate((y, -x, 
                                np.zeros((ptNum, 1)),
                                np.ones((ptNum, 1))),axis=1)
                ))
        
        u = uv[:, 0].reshape(ptNum, 1)
        v = uv[:, 1].reshape(ptNum, 1)
        U = np.concatenate((u,v), axis=0)

        ### X*r = U, Solve the r by least squared error
        if np.linalg.matrix_rank(X)>=2*minNonCollinearPairs:
            r = np.linalg.lstsq(X,U)[0]
        else:
            raise Exception("At least 2 noncollinear Pts")
        
        sc, ss, tx, ty = r
        Tinv = np.array(((sc, -ss, 0),
                         (ss,  sc, 0),
                         (tx,  ty, 1)))
        Tinv = np.linalg.lstsq(normMatDst ,
                               np.dot(Tinv, normMatSrc))[0]
        T = np.linalg.inv(Tinv)
        T[:, 2] = [0,0,1]
        return T

    @classmethod
    def normalizeControlPoints(cls, pts):
        ptNum = pts.shape[0]
        cent  = np.mean(pts, axis=0)
        ptsNorm = np.subtract(pts, cent)
        distSum = np.sum(np.power(ptsNorm, 2))
        if distSum > 0:
            scaleFactor = np.sqrt(2*ptNum)/np.sqrt(distSum)
        else:
            scaleFactor = 1
        
        ptsNorm = scaleFactor * ptsNorm
        normMatInv = np.array(((1/scaleFactor, 0, 0),
                               (0, 1/scaleFactor, 0),
                               (cent[0], cent[1], 1)))
        return ptsNorm, normMatInv
                               

def main():
    src=np.array(((12, 32),
                  (43, 53),
                  (23, 61),
                  (12, 97)))
    dst1=np.array(((12, 32),
                   (43, 53),
                   (23, 61),
                   (12, 97)))
    dst2=np.array(((43, 38),
                   (27, 3),
                   (69, 16),
                   (12, 79)))    

    print("Experiment 1:")
    print Affine.fitGeoTrans(src, dst1)
    print("Experiment 2:")
    print Affine.fitGeoTrans(src, dst2)

    shape = np.array(((1,3),(2,3),(5,4)))
    print(ShapeAug.scale(shape))
    
if __name__ == "__main__":
    main()
