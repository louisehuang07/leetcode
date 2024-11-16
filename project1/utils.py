import numpy as np

def resize(original_image, new_height, new_width):
    "Resize to New Height and Width with Bilinear Neighbor Interpolation"
    height, width = original_image.shape[:2]

    # target image
    resized_image = np.zeros((new_height, new_width, original_image.shape[2]), dtype=original_image.dtype)

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
    bgr_image = np.stack([gray_image] * 3, axis=-1)
    return bgr_image

