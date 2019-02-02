# USAGE
# python test_grader.py --image images/test_01.png

# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
        help="path to the input image")
args = vars(ap.parse_args())
print(args['image'])
# define the answer key which maps the question number
# to the correct answer
#ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1 }
#for multiple choice correct answers
ANSWER_KEY = {0:[1], 1:[2,4], 2:[0], 3:[3], 4:[0,3]}

# load the image, convert it to grayscale, blur it
# slightly, then find edges
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)
#cv2.imshow("edged",edged)
#cv2.waitKey(0)

# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = None

# ensure that at least one contour was found
if len(cnts) > 0:
        # sort the contours according to their size in
        # descending order
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # loop over the sorted contours
        for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                # if our approximated contour has four points,
                # then we can assume we have found the paper
                if len(approx) == 4:
                        docCnt = approx
                        break
                

# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))
#cv2.imshow("paper",paper)
#cv2.imshow("warped",warped)

# apply Otsu's thresholding method to binarize the warped
# piece of paper
thresh = cv2.threshold(warped, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# find contours in the thresholded image, then initialize
# the list of contours that correspond to questions
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
questionCnts = []

# loop over the contours
for c in cnts:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
                questionCnts.append(c)

# sort the question contours top-to-bottom, then initialize
# the total number of correct answers
questionCnts = contours.sort_contours(questionCnts,
        method="top-to-bottom")[0]
#print("length of qustioncnts=",len(questionCnts))
correct = 0

# each question has 5 possible answers, to loop over the
# question in batches of 5
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        # sort the contours for the current question from
        # left to right, then initialize the index of the
        # bubbled answer
        cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
        ans=[]
        bubbled = (None,ans)
        count=0

        # loop over the sorted contours
        for (j, c) in enumerate(cnts):
                # construct a mask that reveals only the current
                # "bubble" for the question
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)

                # apply the mask to the thresholded image, then
                # count the number of non-zero pixels in the
                # bubble area
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                # if the current total has a larger number of total
                # non-zero pixels, then we are examining the currently
                # bubbled-in answer
                #cv2.imshow("masked",mask)
                #cv2.waitKey(0)
                if total > 500:
                        count=count+1;
                if bubbled[0] is None:
                        ans.clear()
                        bubbled=(total,ans)
                        #print(total,bubbled)
                if total > 500:
                        ans.append(j)
                        bubbled=(total,ans)
                        #print(total,bubbled)
        # initialize the contour color and the index of the
        # *correct* answer
        #for single choice correct 
        #if bubbled[0]<500 or count>1:
                #bubbled=(total,-1)
        #for more than one choice correce
        if bubbled[0]<500:
                ans.append(-1)
                bubbled=(total,ans)
        color = (0, 0, 255)
        k = ANSWER_KEY[q]
        print(k,bubbled)

        # check to see if the bubbled answer is correct
        error=0
        for i in range(0,len(k)):
                if k[i] != bubbled[1][i]:
                        error=1;
        if(error==0):
                color = (0, 255, 0)
                correct += 1

        # draw the outline of the correct answer on the test
        for i in range(0,len(k)):
                cv2.drawContours(paper, [cnts[k[i]]], -1, color, 3)
# grab the test taker
score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)
