import cv2

desired_size = 28
color = [255, 255, 255]

def preprocess(im , thresh=1):
    old_size = im.shape[:2] # old_size is in (height, width) format
    target_size = desired_size
    if thresh < 1 and thresh > 0.5 :
        target_size = int(desired_size*thresh)


    ratio = float(target_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    #im = cv2.equalizeHist(im)
    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im


"""
im_pth = "data/0/1471.png"
im = cv2.imread(im_pth, 0)
new_im = preprocess(im)
cv2.imshow("image", new_im)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
