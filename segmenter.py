"""
image to words segmentation for handwriting prediction
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import skimage.color
import skimage.filters
from binary import binary




def segmenter(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    height,width,channel=img.shape
    #resizing for better optimization and speed
    if width>800:
        new_width=800
        aspect_ratio=width/height
        new_height=int(new_width/aspect_ratio)
        img=cv2.resize(img,(new_width,new_height),interpolation=cv2.INTER_AREA)
    

    binary_image=binary(img)

    #below line is for sentences
    kernel = np.ones((3,85), np.uint8)
    dilated_sent= cv2.dilate(binary_image, kernel, iterations = 1)
    #plt.imshow(dilated_sent,cmap="gray")
    #plt.show()
    #below line is for words
    kernal=np.ones((3,15),np.uint8)
    dilated_word=cv2.dilate(binary_image,kernal,iterations=1)

    contour, _ = cv2.findContours(dilated_sent.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    sorted_lines=sorted(contour,key=lambda ctr:cv2.boundingRect(ctr)[1])

    #print(sorted_lines)

    img_copy = img.copy()
    words = []

    for line in sorted_lines:
        x, y, w, h = cv2.boundingRect(line)
        word_line = dilated_word[y:y+w, x:x+w]
        

        (cnt_word, _) = cv2.findContours(word_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        sorted_words = sorted(cnt_word, key=lambda cntr : cv2.boundingRect(cntr)[0])
        
        for word in sorted_words:
            
            if cv2.contourArea(word) < 200:
                continue
            
            x2, y2, w2, h2 = cv2.boundingRect(word)
            words.append([x+x2, y+y2, x+x2+w2, y+y2+h2])
            fin_image=cv2.rectangle(img_copy, (x+x2, y+y2), (x+x2+w2, y+y2+h2), (255,255,100),2)

    #final image with words highlightend   
    plt.imshow(fin_image)
    plt.show() 

    #sorting in y axis
    words.sort(key=lambda w:w[1])

    #getting the unique list
    unique_list = []
    seen = set()
    for x in words:
        if tuple(x) not in seen:
            seen.add(tuple(x))
            unique_list.append(x)
    print(len(unique_list))
    

    #again sorting to get words in orderly manner
    """ initial_y = unique_list[0][1]
    curr_line = []
    group = []

    for i in range(len(unique_list)):
        if abs(unique_list[i][1] - initial_y) < 20:
            curr_line.append(unique_list[i])
        else:
            curr_line.sort(key=lambda w: w[0])
            group.append(curr_line)
            curr_line = [unique_list[i]]
            initial_y = unique_list[i][1]


    flat_group = []
    for line in group:
        for box in line:
            flat_group.append(box) """



    initial_y = unique_list[0][1]
    curr_line = []
    group = []
    line_indices = [] 

    for i in range(len(unique_list)):
        if abs(unique_list[i][1] - initial_y) < 20:
            curr_line.append(unique_list[i])
        else:
            curr_line.sort(key=lambda w: w[0])
            group.append(curr_line)
            line_indices.append([unique_list.index(w) for w in curr_line]) # save indices of words in this line
            curr_line = [unique_list[i]]
            initial_y = unique_list[i][1]

    # append last line
    if len(curr_line) > 0:
        curr_line.sort(key=lambda w: w[0])
        group.append(curr_line)
        line_indices.append([unique_list.index(w) for w in curr_line])

    # flatten the group list and restore the original order of the words
    flat_group = [unique_list[i] for line in line_indices for i in line]

    word_images=[]
    print("2",len(flat_group))
    for i in range(len(flat_group)):
        imag = flat_group[i]
        word = img[imag[1]:imag[3], imag[0]:imag[2]]
        word_images.append(word)
    
    return word_images



""" img=cv2.imread("./word_segmenter/image2.jpg")
images=segmenter(img)
plt.imshow(images[51])
plt.show() """