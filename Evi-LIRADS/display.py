def add_contour_to_img(img_array, mask_array, iscolor, color, linewidth):
    import numpy as np
    import copy
    import cv2
    import matplotlib.pyplot as plt

    # sizeX, sizeY = img_array.shape
    # mask_array_resized = cv2.resize(mask_array,(sizeY, sizeX))
    img_array_contour = copy.deepcopy(img_array)
    if np.max(mask_array) > 255: # for mask is not binary, like int64 array with gray value > 255
        mask_array_normalized = copy.deepcopy(mask_array)
        cv2.normalize(mask_array, mask_array_normalized, 0, 255, cv2.NORM_MINMAX)
        img = np.uint8(mask_array_normalized)
    else:
        img = np.uint8(mask_array)


    ret, binary = cv2.threshold(img,0,1,cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if iscolor:
        cv2.normalize(img_array_contour, img_array_contour, 0, 255, cv2.NORM_MINMAX)
        img_array_contour = np.uint8(img_array_contour)
        img_array_contour = cv2.cvtColor(img_array_contour, cv2.COLOR_GRAY2BGR)

    # plt.imshow(img_array_contour)
    # plt.imshow(img_array_contour_RGB)

    cv2.drawContours(img_array_contour, contours, -1, color, thickness=linewidth) # -1表示绘制所有轮廓; 不填充绘制轮廓：thickness=linewidth，cv2.FILLED

    # plt.imshow(img_array_contour_RGB)
    # plt.show()


    # boderPoint, Distance = Get_BoderPoint(contours, sizeX, sizeY)
    #
    # cv2.drawContours(img_array_contour, boderPoint, -1, (255, 255, 255), 100)  # default bold:3
    #
    # # cv2.drawContours(img_array_contour, RectPointRight, -1, (255, 255, 255), 50)  # default bold:3
    # # cv2.drawContours(img_array_contour, RectPointLeft, -1, (255, 255, 255), 50)  # default bold:3
    #
    # RectPoint, theta = Get_MinAreaRect_from_Binary_Image(binary)
    #
    # color, positioningIssue = Get_iQC_Result(Distance, theta)
    #
    # cv2.drawContours(img_array_contour, RectPoint, -1, (0, 0, 65536), 50)  # default bold:3

    return img_array_contour




def gray2RGB(array):
    import numpy as np
    array_RGB = np.zeros([array.shape[0], array.shape[1], 3])
    array_RGB[:, :, 0] = array
    array_RGB[:, :, 1] = array
    array_RGB[:, :, 2] = array

    return array_RGB

def points_from_contour(contour):
    contour_x = []
    contour_y = []

    for i in range(0,len(contour[0])):
        contour_x.append(contour[0][i][0][0])
        contour_y.append(contour[0][i][0][1])

    return contour_x, contour_y

def drawContour(image_array, liver_array, tumor_array):
    import cv2
    import numpy as np
    #import SimpleITK as sitk
    from matplotlib import pyplot as plt

    image_array = np.squeeze(image_array)
    #image_array = image_array.astype(np.float32)
    image_array = image_array.astype(np.float64)

    plt.figure()
    plt.imshow(image_array, cmap='gray')
    plt.show()

    # windowing 操作
    # min:-200, max:200
    # img = (img-min)/(max - min)
    # image_array = (image_array - (-200)) / 400.0
    # image_array[image_array > 1] = 1.0
    # image_array[image_array < 0] = 0.0
    # 不必须转化为0-255
    cv2.normalize(image_array, image_array, 0, 255, cv2.NORM_MINMAX)


    # 若不转化为彩色，那么最后画出来的contour也只能是灰度的

    image_array = np.uint8(image_array)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

    liver_array = np.squeeze(liver_array)
    # liver_array *= 255 不必须

    tumor_array = np.squeeze(tumor_array)
    # tumor_array *= 255 不必须

    # findContours 必须目标是white，背景是black (0,1 和 0,255 都可以)
    # py3 只返回2个参数，而非之前的3个
    liver_array = np.uint8(liver_array)
    ret, liver_array_binary = cv2.threshold(liver_array, 0, 1, cv2.THRESH_BINARY)
    binary_, contours, hierarchy = cv2.findContours(
        liver_array_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tumor_array = np.uint8(tumor_array)
    ret, tumor_array_binary = cv2.threshold(tumor_array, 0, 1, cv2.THRESH_BINARY)
    binary_, contours2, hierarchy2 = cv2.findContours(
        tumor_array_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    # drawContours 直接改变原图
    # 第三个参数 index
    # 第四个参数color: BGR
    # 两种不同的方式指明画哪个contour
    cv2.drawContours(image_array, [cnt], 0, (0, 0, 255), 1)
    cv2.drawContours(image_array, contours2, -1,
                     (255, 0, 0), 1)  # index=-1表示画所有的contour
    cv2.imshow("liver_contour", image_array)
    cv2.waitKey()

