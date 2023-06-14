# 1 加入库
import cv2
import dlib
import numpy as np

# 2 读取一张图片
im1 = cv2.imread("FacePhoto/14.jpg")
im2 = cv2.imread("FacePhoto/22.jpg")
alpha = 0.5

# 3 调用人脸检测器获取人脸关键点
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


# 定义了一个人脸关键点检测函数
def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


# 获得两张人脸的关键点
landmarks1 = get_landmarks(im1)  # 获的人脸检测点
landmarks2 = get_landmarks(im2)  # 获的人脸检测点


def show_face_detections(im):
    landmarks_im = get_landmarks(im)
    for i in range(landmarks1.shape[0]):  # 所有关键点
        array = landmarks_im[i]
        cv2.circle(im, (array[0, 0], array[0, 1]), 3, (0, 255, 0), -1)


# show_face_detections(im1)
# show_face_detections(im2)
# cv2.imshow("im1_landmark", im1)
# cv2.imshow("im2_landmark", im2)
# cv2.waitKey(0)


def transformation_from_points(points1, points2):  # 得到的是将point2变换到point1的矩阵
    points1 = points1.astype(np.float64)  # 转为float64
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)  # 求得样本点的均值
    c2 = np.mean(points2, axis=0)
    points1 -= c1  # 减去均值，平移 归一化
    points2 -= c2

    s1 = np.std(points1)  # 求得样本点的方差
    s2 = np.std(points2)
    points1 /= s1  # 除以方差  缩放 归一化
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)  # 两个形状的协方差矩阵，进行奇异值分解
    R = (U * Vt).T  # 得到变换矩阵

    return np.vstack([np.hstack(((s2 / s1) * R,
                                 c2.T - (s2 / s1) * R * c1.T)),
                      np.matrix([0., 0., 1.])])


def warp_im(im, M, dshape):  # 用矩阵变换图像im2到im1
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


M = transformation_from_points(landmarks1, landmarks2)
im2 = warp_im(im2, M, im1.shape)
# show_face_detections(im2)
# cv2.imshow("image_transform", im2)  # 变换后的第二张脸
# cv2.waitKey(0)
landmarks2 = get_landmarks(im2)


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def draw_delaunay(img, subdiv, delaunay_color, dictionary1):
    triangleList = subdiv.getTriangleList()
    size = img.shape
    list4 = []
    r = (0, 0, size[1], size[0])
    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            # cv2.line(img, pt1, pt2, delaunay_color, 1, 0)
            # cv2.line(img, pt2, pt3, delaunay_color, 1, 0)
            # cv2.line(img, pt3, pt1, delaunay_color, 1, 0)
            list4.append((dictionary1[pt1], dictionary1[pt2], dictionary1[pt3]))
    return list4


def applyAffineTransform(src, srcTri, dstTri, size):
    # 给定一对三角形，找到仿射变换
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # 将仿射变换应用于src图片
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def morphTriangle(img1, img2, img, t1, t2, t, alpha):
    # 找到每个三角形区域的包络矩形
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # 各个矩形左上角的偏移点
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # 填充三角形来获得掩码
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1, 0), 16, 0)

    # 将warpImage应用于小矩形块
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha混合矩形补丁
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # 将矩形块的三角形区域复制到输出图像
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask


size = im1.shape
rect = (0, 0, size[1], size[0])
subdiv = cv2.Subdiv2D(rect)
points = []
points1 = [[1, 1], [size[1] - 1, 1], [size[1] // 2, 1], [1, size[0] - 1], [1, size[0] // 2 + 1],
           [size[1] - 1, size[0] - 1], [size[1] // 2 + 1, size[0] - 1],
           [size[1] - 1, size[0] // 2]] + landmarks1.tolist()
points2 = [[1, 1], [size[1] - 1, 1], [size[1] // 2, 1], [1, size[0] - 1], [1, size[0] // 2 + 1],
           [size[1] - 1, size[0] - 1], [size[1] // 2 + 1, size[0] - 1],
           [size[1] - 1, size[0] // 2]] + landmarks2.tolist()

for i in range(0, 76):
    x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
    y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
    points.append((x, y))
# 修改一下格式
points = [(int(x[0]), int(x[1])) for x in points]
points1 = [tuple(element) for element in points1]
points2 = [tuple(element) for element in points2]

# insert point into subdiv
for point in points:
    subdiv.insert((point[0], point[1]))
dictionary1 = {point[0]: point[1] for point in list(zip(points, range(76)))}
trianglepoint = draw_delaunay(im2, subdiv, (255, 255, 255), dictionary1)
# cv2.imshow("im1_delaunay", im2)
# cv2.waitKey(0)

# 为最后的输出分配空间
imgMorph = np.zeros(im1.shape, dtype=im1.dtype)

for line in trianglepoint:
    x = line[0]
    y = line[1]
    z = line[2]

    t1 = [points1[x], points1[y], points1[z]]
    t2 = [points2[x], points2[y], points2[z]]
    t = [points[x], points[y], points[z]]

    # 一次合成一个三角形
    morphTriangle(im1, im2, imgMorph, t1, t2, t, alpha)

# 输出结果
# cv2.imshow("1.jpg", im1)
# cv2.imshow("2.jpg", im2)
# cv2.imshow("warp_face", np.uint8(imgMorph))
# cv2.waitKey(0)


# 获得模板
def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)
    points = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(im, points, color=1)
    im = np.array([im, im, im]).transpose((1, 2, 0))
    return im


def find_center(img):
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 获取每个轮廓外接矩形的左上角坐标和宽度高度
    rects = [cv2.boundingRect(c) for c in contours]
    # 计算每个矩形的中心点坐标
    centers = [(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in rects]
    # 计算整个图像的几何中心
    center = np.mean(centers, axis=0)
    return center


# 获得遮罩mask
points = np.array(points[-68:])
mask = get_face_mask(imgMorph, points)
# 裁剪之后直接拼接
# Mask_im1 = im1 * (1.0-mask)
# Mask_im1 = np.clip(Mask_im1, 0, 255).astype(np.uint8)
# Mask_im2 = imgMorph * mask
# Mask_im2 = np.clip(Mask_im2, 0, 255).astype(np.uint8)  # 调整到（0，255）范围
# cv2.imshow("1", Mask_im1)
# # cv2.imshow("part2", Mask_im2)
# res = cv2.add(Mask_im1, Mask_im2)
# cv2.imshow("Morphed Face.jpg", res)

# seamless融合
Mask1 = cv2.convertScaleAbs(im1)
Mask2 = cv2.convertScaleAbs(imgMorph)
mask = cv2.convertScaleAbs(mask, alpha=255 / np.max(mask))
center = find_center(mask)
x = int(center[0])
y = int(center[1])
respond = cv2.seamlessClone(Mask2, Mask1, mask, (x, y), cv2.NORMAL_CLONE)
# cv2.imshow("part1", Mask1)
# cv2.imshow("part2", Mask2)
# cv2.imshow("image_result222", respond)
images = np.hstack([im1, respond, im2])
cv2.imshow("1", images)
cv2.waitKey(0)
