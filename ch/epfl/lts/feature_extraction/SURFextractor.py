import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def main():
    indir = 'D:\cours\MA1\Semester Project\datasets\holidays\jpg'
    outdir = 'D:\cours\MA1\Semester Project\datasets\holidays\jpg-surf'
    for root, dirs, filenames in os.walk(indir):
        for f in filenames:
            infile = os.path.join(root,f)
            outfile = os.path.join(outdir,os.path.splitext(f)[0]+'-surf128')
            #print infile
            #print outfile
            print f
            extractAndStore(infile, outfile)


def extractAndStore(infile, outfile):
    img = cv2.imread(infile,0)
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    surf = cv2.SURF(700)
    surf.extended = True
    surf.upright = True
    kp, desc = surf.detectAndCompute(img,None)
    # while not np.abs(len(kp)-300)<100 :
    #     surf.hessianThreshold += 500
    #     kp, desc = surf.detectAndCompute(img, None)
    #img2 = cv2.drawKeypoints(img,kp,None, (255,0,0),4)

    np.save(outfile,desc)


main()