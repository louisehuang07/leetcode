import math
import numpy as np
import cv2
def get8n(x, y, shape):
    """
    get 8 coordinates of position (x, y)

    paramaters: 
        x (int): row
        y (int): col
        shape (tuple): image size (height, width)。

    return:
        neighbors (list): 8 邻域的坐标列表。
    """
    neighbors = []
    for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        xn, yn = x + dx, y + dy
        if 0 <= xn < shape[0] and 0 <= yn < shape[1]:  # 排除越界点
            neighbors.append((xn, yn))
    return neighbors

def hough_transformer(image, threshold_number):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # dual threshold
    mask = cv2.inRange(gray, 0, 255)
    result = cv2.bitwise_and(gray, gray, mask=mask)

    # deblur with open operation
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    # Erosion
    num_iterations = 2  # Erosion time
    for i in range(num_iterations):
        opening = cv2.erode(opening, kernel, iterations=1)

    # Detect dege
    edges = cv2.Canny(opening, 200, 250)

    # Detect image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold_number, minLineLength=70, maxLineGap=30)
    print('lines,\n',lines)
    print('threshold_number,\n',threshold_number)

    # convert
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # If Detect image, show angles
    if lines is not None:
        previous_angle = None  # store angle
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 2)  # draw line
            
            # angle of line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # only if angle > 1
            if previous_angle is None or abs(angle - previous_angle) > 1:
                # add label
                cv2.putText(gray, f"Angle: {angle:.2f} degrees", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                previous_angle = angle

    return gray
    


def erosion_and_diliation_operator(image):
    # kernel
    kernel = np.ones((3,3), np.uint8) 

    # erosion
    erosion_result = cv2.erode(image,kernel, iterations = 1)

    # kernel 2
    kernel_2 = np.ones((3,3),np.uint8) 

    # dilate
    dilation_result = cv2.dilate(image, kernel_2, iterations = 1)

    # extract edge
    # return cv2.absdiff(dilation_result, erosion_result)
    return dilation_result

def erosion_and_diliation_operator_(image):
    # convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # kernel
    kernel = np.ones((3,3), np.uint8) 

    # erosion
    erosion_result = erodision(image,kernel)

    # kernel 2
    kernel_2 = np.ones((3,3),np.uint8) 

    # dilate
    dilation_result = dilation(image, kernel_2)

    # extract edge
    image = np.abs(np.subtract(dilation_result, erosion_result))
    
    return image

def erodision(image, kernel):
    # center of kernel
    # kernel_center = tuple(x//2 for x in kernel.shape)
    # kernel shape 
    h, w = kernel.shape
    # pad shape
    pad_h, pad_w = h // 2, w // 2
    # pad image with constant mode
    print(image.shape)  # 查看图像形状
    print(kernel.shape)  # 查看结构元素形状

    image_padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    print('image_padded',image.shape)  # 查看图像形状
    
    # shape like origin image
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = image_padded[i:i + h, j:j + w]
            if np.all(region[kernel == 1] > 0): # if value within this region is not 0
                output[i, j] = 255 # np.max(region)
            else:
                output[i, j] = 0  # black
    return output

def dilation(image, kernel):
    # shape like origin image
    output = np.zeros_like(image)
    # pad image with constant mode
    h, w = kernel.shape
    pad_h, pad_w = h // 2, w // 2  
    image_padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = image_padded[i:i + h, j:j + w]
            if np.any(region[kernel == 1] > 0):
                output[i, j] = 255 # np.max(region)
            else:
                output[i, j] = 0  # black
    return output

def canny_operator(image):
    
    return cv2.Canny(image, 120, 250)


def canny_operator_(image):
    # convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. smooth image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # gradient 
    sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    # magnitude
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # direction
    gradient_direction = np.arctan2(sobely, sobelx)

    # non maximize suppression
    nms_output = non_max_suppression(gradient_magnitude, gradient_direction)
    
    # dual threshold
    low_threshold = 0.05 * np.max(nms_output)
    high_threshold = 0.15 * np.max(nms_output)

    strong_edges, weak_edges = double_threshold(nms_output, low_threshold, high_threshold)
    final_edges = edge_tracking(strong_edges, weak_edges)

    return final_edges

def non_max_suppression(gradient_magnitude, gradient_direction):
    h, w = gradient_magnitude.shape
    nms = np.zeros((h, w), dtype=np.uint8)
    
    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            try:
                q = 255
                r = 255
                
                # angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = gradient_magnitude[i, j+1]
                    r = gradient_magnitude[i, j-1]
                # angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = gradient_magnitude[i+1, j-1]
                    r = gradient_magnitude[i-1, j+1]
                # angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = gradient_magnitude[i+1, j]
                    r = gradient_magnitude[i-1, j]
                # angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = gradient_magnitude[i-1, j-1]
                    r = gradient_magnitude[i+1, j+1]

                if (gradient_magnitude[i,j] >= q) and (gradient_magnitude[i,j] >= r):
                    nms[i,j] = gradient_magnitude[i,j]
                else:
                    nms[i,j] = 0

            except IndexError as e:  
                pass
                
    return nms


def double_threshold(nms, low_threshold, high_threshold):
    strong_i, strong_j = np.where(nms >= high_threshold)
    weak_i, weak_j = np.where((nms <= high_threshold) & (nms >= low_threshold))
    strong_edges = np.zeros(nms.shape, dtype=np.uint8)
    strong_edges[strong_i, strong_j] = 255
    weak_edges = np.zeros(nms.shape, dtype=np.uint8)
    weak_edges[weak_i, weak_j] = 255
    return strong_edges, weak_edges

def edge_tracking(strong_edges, weak_edges):
    final_edges = strong_edges.copy()
    def track(i, j):
        if weak_edges[i, j] != 0:
            final_edges[i, j] = 255
            weak_edges[i, j] = 0
            neighbors = [(i-1, j-1), (i-1, j), (i-1, j+1),
                         (i, j-1),           (i, j+1),
                         (i+1, j-1), (i+1, j), (i+1, j+1)]
            
            for ni, nj in neighbors:
                if 0 <= ni < weak_edges.shape[0] and 0 <= nj < weak_edges.shape[1]:
                    track(ni, nj)
    for i in range(final_edges.shape[0]):
        for j in range(final_edges.shape[1]):
            if final_edges[i, j] == 255:
                track(i, j)
    return final_edges




def apply_gaussian(image, kernel_size=5, sigma=1.4):
    kernel = gaussian_filter(kernel_size, sigma)
    return cv2.filter2D(image, -1, kernel)  
def gaussian_filter(kernel_size=5, sigma=1.4):
    """
    Generate a gaussian filter
    """
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)  

def compute_gradient(image):
    """
    Compute gradient and direction of image
    """
    # Sobel for gradient
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    grad_x = cv2.filter2D(image, -1, sobel_x)
    grad_y = cv2.filter2D(image, -1, sobel_y)
    
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)  # magnitude
    grad_dir = np.arctan2(grad_y, grad_x)  # direction
    return grad_mag, grad_dir

def non_maximum_suppression(grad_mag, grad_dir):
    """
    Conduct non maximum suppression
    """
    h, w = grad_mag.shape
    suppressed = np.zeros_like(grad_mag)
    
    # gradient to angle
    grad_dir = np.rad2deg(grad_dir) % 180
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            angle = grad_dir[i, j]
            mag = grad_mag[i, j]
            
            # compare
            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                # left and right
                neighbors = [grad_mag[i, j-1], grad_mag[i, j+1]]
            elif 22.5 <= angle < 67.5:
                # rift_up and right_down
                neighbors = [grad_mag[i-1, j-1], grad_mag[i+1, j+1]]
            elif 67.5 <= angle < 112.5:
                # up and down
                neighbors = [grad_mag[i-1, j], grad_mag[i+1, j]]
            else:
                # rift_down and right_up
                neighbors = [grad_mag[i-1, j+1], grad_mag[i+1, j-1]]
            
            # non maximum suppression
            if mag >= max(neighbors):
                suppressed[i, j] = mag
            else:
                suppressed[i, j] = 0
    
    return suppressed

def double_thresholding(suppressed, low_threshold, high_threshold):
    """
    Double thresholding
    """
    strong_edges = suppressed > high_threshold
    weak_edges = (suppressed >= low_threshold) & (suppressed <= high_threshold)
    return strong_edges, weak_edges

def edge_tracking(strong_edges, weak_edges):
    """
    Edge link to preserve meaningful edge
    """
    h, w = strong_edges.shape
    edges = strong_edges.copy()

    for i in range(1, h-1):
        for j in range(1, w-1):
            if weak_edges[i, j]:
                # connect
                if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                    edges[i, j] = 1  
                else:
                    edges[i, j] = 0  

    return edges




def resize(original_image, scale_factor):
    "Resize to New Height and Width with Bilinear Neighbor Interpolation"
    new_width = int(original_image.shape[1] * scale_factor)
    new_height = int(original_image.shape[0] * scale_factor)

    height, width = original_image.shape[:2]

    # target image
    resized_image = np.zeros((new_height, new_width, original_image.shape[2]), 
                             dtype=original_image.dtype)

    # ratio
    ratio_row = height / new_height
    ratio_col = width / new_width

    for i in range(new_height):
        for j in range(new_width):
            # original 4 coords
            original_i = i * ratio_row
            original_j = j * ratio_col
            
            # nearby 4 coords
            x1 = int(original_i)
            y1 = int(original_j)
            x2 = min(x1 + 1, height - 1)
            y2 = min(y1 + 1, width - 1)
            
            # weight
            alpha = original_i - x1
            beta = original_j - y1
            
            # nearby 4 pixel
            I_x1_y1 = original_image[x1, y1]
            I_x2_y1 = original_image[x2, y1]
            I_x1_y2 = original_image[x1, y2]
            I_x2_y2 = original_image[x2, y2]
            
            # bilinear
            resized_image[i, j] = (1 - alpha) * (1 - beta) * I_x1_y1 + alpha * (1 - beta) * I_x2_y1 + (1 - alpha) * beta * I_x1_y2 + alpha * beta * I_x2_y2
            
    return resized_image


def rotate(image):
    "Rotate with numpy"
    image = np.array(image)
    image = np.transpose(image, (1, 0, 2))
    image = np.fliplr(image)
    return image

def rotate_1(image):
    "Rotate with 90° Clockwise."
    image_rot = np.array(image)
    assert image_rot.ndim == 3 and image_rot.shape[2] == 3
    # size of image
    (h, w) = image.shape[:2]
    
    # center of image
    center = (w // 2, h // 2) 

    # rotate function TODO
    angle = 90
    angle_rad = np.deg2rad(angle)  
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)

    rotation_matrix = np.array([
        [cos_val, -sin_val, center[0] - cos_val * center[0] + sin_val * center[1]],
        [sin_val, cos_val, center[1] - sin_val * center[0] - cos_val * center[1]]
    ])
    print('rotation_matrix',rotation_matrix)
    # empty image
    
    rotated_image = np.zeros_like(image)

    for y in range(h):
        for x in range(w):
            # print('image.shape', image.shape)
            # print('h', h)
            # print('w', w)
            new_x = int(rotation_matrix[0, 0] * x + rotation_matrix[0, 1] * y + rotation_matrix[0, 2])
            # print('x',x)
            # print('new_x', new_x)
            new_y = int(rotation_matrix[1, 0] * x + rotation_matrix[1, 1] * y + rotation_matrix[1, 2])
            # print('new_y', new_y)
            # print('y',y)
            
            if 0 <= new_x < h and 0 <= new_y < w:
                rotated_image[new_y, new_x] = image[x, y]

    return image
    

def bitwise_not(image):
    "Invert Colors."
    inverted_image = np.array(image)
    assert inverted_image.ndim == 3 and inverted_image.shape[2] == 3

    # bitwise inversion
    inverted_image = ~inverted_image
    return inverted_image

def grayscale(image):
    "From (h,w,3)rgb to (h,w)grayscale to (h,w,3)grayscale."
    image_rgb = np.array(image)
    assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3

    # extract rgb
    R, G, B = image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]

    # weighted average
    gray_image = 0.299 * R + 0.587 * G + 0.114 * B
    bgr_image = np.stack([gray_image] * 3, axis=-1).astype(np.uint8)
    # print('image_rgb',image_rgb.shape)  # (h,w,3)
    # print('gray_image',gray_image.shape)    # (h,w)
    # print('bgr_image',bgr_image.shape) # (h,w,3)
    return bgr_image

def grayscale_single_channel(image):
    image_rgb = np.array(image)
    assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3

    # extract rgb
    R, G, B = image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]

    # weighted average
    gray_image = 0.299 * R + 0.587 * G + 0.114 * B
    return gray_image

def check_if_image_exist(image):
    if image is None:
        return 
    
def warpAffine_m(image, shear_angle):
    h,w = image.shape[:2]

    # tan   
    tan = math.tan(math.radians(shear_angle))

    # affine transform matrix_1
    shear_matrix = np.array([
        [1, tan, 0],
        [0, 1, 0]
    ], dtype=np.float32)
    
    # origin coordinate
    origin_coords = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]])
    
    # shear coordinate 
    shear_coords = (shear_matrix @ origin_coords.T).T.astype(int)


    # new size of image
    new_w = int(np.ceil(np.max(shear_coords[:, 0]) - np.min(shear_coords[:, 0])))
    new_h = int(np.ceil(np.max(shear_coords[:, 1]) - np.min(shear_coords[:, 1])))
    
    # off set
    x_offset = -int(np.floor(np.min(shear_coords[:, 0])))
    y_offset = -int(np.floor(np.min(shear_coords[:, 1])))
    
    # initialize result image
    result = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    
    # inv.shear_matrix
    inv_shear_matrix = np.linalg.inv(
        np.vstack([shear_matrix, [0, 0, 1]])
    )[:2, :]  
    
    # every pixel
    for y_new in range(new_h):
        for x_new in range(new_w):
            # map: result to origin
            src_coords = np.dot(inv_shear_matrix, [x_new - x_offset, y_new - y_offset, 1])
            x_orig, y_orig = src_coords[0], src_coords[1]
            
            # if within origin image
            if 0 <= x_orig < w and 0 <= y_orig < h:
                # nearest 4 points
                x1, y1 = int(np.floor(x_orig)), int(np.floor(y_orig))
                x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
                
                # weight of bilinear
                a, b = x_orig - x1, y_orig - y1
                
                # pixel value
                pixel = (
                    (1 - a) * (1 - b) * image[y1, x1] +
                    a * (1 - b) * image[y1, x2] +
                    (1 - a) * b * image[y2, x1] +
                    a * b * image[y2, x2]
                )
                
                # assign value
                result[y_new, x_new] = np.clip(pixel, 0, 255)
    
    return result

def perspective_m(image):
    # original position
    src_points = np.float32([
        [100, 100],  # left-up
        [400, 100],  # right-up
        [100, 400],  # left-up
        [400, 400]   # left-down
    ])

    # target position
    dst_points = np.float32([
        [50, 50],    # left-up
        [450, 50],   # right-up
        [100, 400],  # left-up
        [400, 400]   # left-down
    ])
    # get perspective matrix
    perspective_matrix = calculate_perspective_matrix(src_points, dst_points)

    # apply perspective transform
    h, w = image.shape[:2]
    result = warp_perspective(image, perspective_matrix, h, w)
    return result 

def calculate_perspective_matrix(src_points, dst_points):
    """
    Reimplement of cv2.getPerspectiveTransform
    param src_points: origin position, shape = (4, 2)
    param dst_points: target position, shape = (4, 2)
    return: 3x3 perspective transform matrix
    """
    A = []
    B = []

    for i in range(4):
        x, y = src_points[i]
        x_prime, y_prime = dst_points[i]
        # A.append([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y, -x_prime])
        # A.append([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y, -x_prime])
        A.append([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y])
        A.append([0, 0, 0, x, y, 1, -y_prime * x, -y_prime * y])
        B.extend([x_prime, y_prime])

    # solve equations A @ h = B
    A = np.array(A, dtype=np.float32)   # 8x8
    B = np.array(B, dtype=np.float32)   # 8x1
    # h = np.linalg.solve(A, B)

    h = np.linalg.lstsq(A, B, rcond=None)[0]

    # to 3x3 matrix
    H = np.append(h, 1).reshape(3, 3)
    return H

def warp_perspective(image, perspective_matrix, h, w):
    """
    Reimplement of cv2.warpPerspective, with bilinear interpolation
    :param image: input image
    :param perspective_matrix
    :param output_size: height and width of output image
    :return: transformed image
    """
    original_h, original_w = image.shape[:2]
    result = np.zeros((h, w, 3), dtype=np.uint8)

    # inv.perspective matrix
    inv_matrix = np.linalg.inv(perspective_matrix)

    for y_new in range(h):
        for x_new in range(w):
            #  target points to origin points
            src_coords = np.dot(inv_matrix, [x_new, y_new, 1])
            x_orig, y_orig = src_coords[0] / src_coords[2], src_coords[1] / src_coords[2]

            # if within image
            if 0 <= x_orig < original_w and 0 <= y_orig < original_h:
                #  bilinear interpolation
                x1, y1 = int(np.floor(x_orig)), int(np.floor(y_orig))
                x2, y2 = min(x1 + 1, original_w - 1), min(y1 + 1, original_h - 1)

                a, b = x_orig - x1, y_orig - y1

                pixel = (
                    (1 - a) * (1 - b) * image[y1, x1] +
                    a * (1 - b) * image[y1, x2] +
                    (1 - a) * b * image[y2, x1] +
                    a * b * image[y2, x2]
                )
                result[y_new, x_new] = np.clip(pixel, 0, 255)

    return result

def equalization(image):
    # 1. calculate histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # 2. calculate cumnulative distribution function(CDF)
    cdf = hist.cumsum()  # calculate the cumulative sum
    # cdf_normalized = cdf * float(hist.max()) / cdf.max()  # normalize CDF

    # 3. use CDF for mapping
    cdf_m = np.ma.masked_equal(cdf, 0)  # avoid division by 0 errors
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # normalize to 0-255
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')  # restore CDF to uint8 type

    # 4. use mapping relationship to map image pixel values to new grayscale values
    image = image.astype('uint8')
    img_eq = cdf[image]

    return img_eq


def median_filter(image):
    kernel_size = 3

    # image size
    height, width, channels = image.shape 

    # check  kernel size 
    if kernel_size %2 == 0:
        raise ValueError("kernel_size must be an odd number.")
    
    # edge
    pad = kernel_size // 2

    # empty output image
    output_image = np.copy(image)
    # each pixel
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            for c in range(channels):
                # nearby window
                window = image[i - pad:i + pad + 1, j - pad:j + pad + 1, c]
                
                # median value
                median_value = np.median(window)
                
                # replace pixel value with median value
                output_image[i, j, c] = median_value
    
    return output_image


def robert_opt(image):
    # def kernel of Robert
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    # calculate the gradient of an image
    height, width = image.shape[:2]
    
    # output image
    output_image = np.zeros_like(image, dtype=np.float32)
    # single channel image to get grad_image
    image_single_channel = grayscale_single_channel(image)
    
    # convolutional operation
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # gradient along x & y
            grad_x = np.sum(kernel_x * image_single_channel[i:i+2, j:j+2])
            grad_y = np.sum(kernel_y * image_single_channel[i:i+2, j:j+2])
            
            # Gradient Amplitude
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Replace image pixel values
            output_image[i, j] = grad_mag
    
    # Convert the result to uint8 type
    grad_image = np.clip(output_image, 0, 255).astype(np.uint8)
    sharpened_image = addWeighted_m(image, grad_image)

    # make sure image in [0,255]
    image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
    return image

# def sobel_opt(image):
#     # def kernel of Robert
#     kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
#     kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    
#     # size of image
#     height, width = image.shape[:2]
    
#     # initialize output image
#     output_image = np.zeros_like(image, dtype=np.float32)
#     # single channel image to get grad_image
#     image_single_channel = grayscale_single_channel(image)

#     # Perform convolution on the image
#     for i in range(1, height - 1):
#         for j in range(1, width - 1):
#             # Calculate the gradient in the x and y directions
#             grad_x = np.sum(kernel_x * image_single_channel[i-1:i+2, j-1:j+2])
#             grad_y = np.sum(kernel_y * image_single_channel[i-1:i+2, j-1:j+2])
            
#             # Gradient Amplitude
#             grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
#             # Replace image pixel values
#             output_image[i, j] = grad_mag
    
#     # Convert the result to uint8 type
#     grad_image = np.clip(output_image, 0, 255).astype(np.uint8)
#     sharpened_image = addWeighted_m(image, grad_image)

#     # make sure image in [0,255]
#     image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
#     return image
def sobel_opt(image):
    # Sobel convolutional kernel
    kernel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1],
                            [0,  0,  0],
                            [1,  2,  1]], dtype=np.float32)

    # image size
    h, w, c = image.shape

    # edge image
    edge_image = np.zeros((h , w , c), dtype=np.uint8)

    # per channel
    for ch in range(c):
        # single channel
        channel = image[:, :, ch]

        # initialize
        grad_x = np.zeros((h , w ), dtype=np.float32)
        grad_y = np.zeros((h , w ), dtype=np.float32)

        # convolution
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                region = channel[i - 1:i + 2, j - 1:j + 2]  # 提取 3x3 区域
                grad_x[i - 1, j - 1] = np.sum(region * kernel_x)  # 水平梯度
                grad_y[i - 1, j - 1] = np.sum(region * kernel_y)  # 垂直梯度

        # gradient, normalize to [0, 255]
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)

        # store to channel
        edge_image[:, :, ch] = gradient_magnitude
    ret = addWeighted_m(image, edge_image)
    return ret


def laplace_opt(image):
    # def kernel of Laplacian
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    
    #  size of image
    height, width = image.shape[:2]
    
    # initialize output image
    output_image = np.zeros_like(image, dtype=np.float32)
    
    # Perform convolution on the image
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # pixel value
            output_image[i, j] = np.sum(kernel * image[i-1:i+2, j-1:j+2])
    
    # Convert the result to uint8 type
    grad_image = np.clip(output_image, 0, 255).astype(np.uint8)
    sharpened_image = addWeighted_m(image, grad_image)

    return sharpened_image

def addWeighted_m(image, grad_image):
    # \alpha * image + \beta * grad_image + \gamma
    alpha = 1
    beta = 1
    gamma = 0
    if image.shape != grad_image.shape:
        print('image.shape=', image.shape)
        print('grad_image.shape=', grad_image.shape)

        raise ValueError("The shape of the two images must match!")
    
    # float 32
    image = image.astype(np.float32)
    grad_image = grad_image.astype(np.float32)
    # linear formular
    result = alpha * image + beta * grad_image + gamma
    result = np.clip(result,0,255).astype(np.uint8)

    return result

