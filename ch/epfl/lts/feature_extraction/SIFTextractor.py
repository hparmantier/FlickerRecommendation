
import numpy as np
import cv2
import os
import math
import binascii

def main():
    indir = 'D:\cours\MA1\Semester Project\datasets\holidays\jpg'
    outdir = 'D:\cours\MA1\Semester Project\datasets\holidays\jpg-sift-lowes'

    for root, dirs, filenames in os.walk(indir):
        for f in filenames:
            infile = os.path.join(root,f)
            outfile1 = os.path.join(outdir,os.path.splitext(f)[0]+'-sift')
            outfile2 = os.path.join(outdir,os.path.splitext(f)[0]+'-lowes.sift')
            #print infile
            #print outfile
            print f
            kp, desc = extractAndStore(infile, outfile1)
            print len(kp)
            print desc.shape
            print kp[0].pt

            saveLoweFormat(kp, desc, outfile2)





def extractAndStore(infile,outfile):
    img = cv2.imread(infile)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
    detector = cv2.SIFT(2000)
    return detector.detectAndCompute(gray, None)


    # print kp[0].pt
    # print kp[0].size
    # print kp[0].angle

    #img_sift = cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imwrite('D:\cours\MA1\Semester Project\img_sift_example.jpg', img_sift)

    #np.save(outfile, desc)

def saveLoweFormat(kp, desc, outfile):
    f = open(outfile, "w")
    f.write('SIFT\n')
    f.write('V4.0\n')
    f.write(str(len(kp)) + ' 128\n')
    for i in range(len(kp)):
        point = kp[i]
        descriptor = map(lambda x: str(x), desc[i])
        x,y = point.pt
        f.write(str(x) + ' ' + str(y) + ' ' + str(point.size) + ' ' + str(math.radians(point.angle)) + '\n')
        f.write(' '.join(descriptor)+'\n')
    f.close()

main()