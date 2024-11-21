import math
import numpy as np
    
def resize(original_image, scale_factor):
    "Resize to New Height and Width with Bilinear Neighbor Interpolation"
    new_width = int(original_image.shape[1] * scale_factor)
    new_height = int(original_image.shape[0] * scale_factor)
    # print('new_height',new_height)
    # print('new_width',new_width)
    # print('scale_factor',scale_factor)

    height, width = original_image.shape[:2]
    # print('height', height)
    # print('width', width)
    # target image
    resized_image = np.zeros((new_height, new_width, original_image.shape[2]), dtype=original_image.dtype)

    # ratio
    ratio_row = height / new_height
    ratio_col = width / new_width
    # print('ratio_row', ratio_row)
    # print('ratio_col', ratio_col)
    # print('scale_factor',scale_factor)
    

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
    )[:2, :]  # 截取前两行
    
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
    
    # convolutional operation
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # gradient along x & y
            grad_x = np.sum(kernel_x * image[i:i+2, j:j+2])
            grad_y = np.sum(kernel_y * image[i:i+2, j:j+2])
            
            # Gradient Amplitude
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Replace image pixel values
            output_image[i, j] = grad_mag
    
    # Convert the result to uint8 type
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    return output_image

def sobel_opt(image):
    # def kernel of Robert
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    
    # size of image
    height, width = image.shape[:2]
    
    # initialize output image
    output_image = np.zeros_like(image, dtype=np.float32)
    
    # Perform convolution on the image
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Calculate the gradient in the x and y directions
            grad_x = np.sum(kernel_x * image[i-1:i+2, j-1:j+2])
            grad_y = np.sum(kernel_y * image[i-1:i+2, j-1:j+2])
            
            # Gradient Amplitude
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Replace image pixel values
            output_image[i, j] = grad_mag
    
    # Convert the result to uint8 type
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    return output_image
    

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
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    return output_image
