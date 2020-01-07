import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
  image = cv2.imread(path)
  return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def mask_to_gray(mask):
  return np.array(mask).astype(np.uint8)*255

def trim_image(img, min_x, max_x, min_y, max_y):
  return img[min_y:max_y, min_x:max_x]

def is_gray(img):
  # gray images have either 1 channel or R = G = B
  if len(img.shape) > 2:
    channels = cv2.split(img)
    if np.array_equal(channels[0], channels[1]) \
      and np.array_equal(channels[0], channels[2]):
      return True
    else:
      return False
  else:
    return True

def display_image(img):
    if is_gray(img):
      plt.imshow(img, cmap="gray")
    else:
      plt.imshow(img)
    plt.show()

def display_images(imgs):
  if len(imgs) == 1:
    display_image(imgs[0])
    return 
  fig, axs = plt.subplots(1, len(imgs))
  for ax, img in zip(axs, imgs):
    if is_gray(img):
      ax.imshow(img, cmap="gray")
    else:
      ax.imshow(img)
  plt.show()

def rotate_image(image, angle):
  """
  Args:
    Image
    Angle - in degrees

  Returns:
    Image rotated by angle
  """
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def alpha_blend(foreground, background, alpha):
  """
  Args:
    foreground - image to be blended in
    background - image to be blended on
    alpha - mask image
  
  Returns:
    Alpha blended image, with float values

  Assumptions:
    Images have same dimension
  """

  # convert uint8 to float
  foreground = foreground.astype(float)
  background = background.astype(float)
 
  # normalize the alpha mask to keep intensity between 0 and 1
  alpha = alpha.astype(float)/255

  # multiply the foreground with the alpha matte
  foreground = cv2.multiply(alpha, foreground)
 
  # multiply the background with ( 1 - alpha )
  background = cv2.multiply(1.0 - alpha, background)
 
  # add the masked foreground and background.
  out = cv2.add(foreground, background)
  return out

def resize_image(img, width, height):
  return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

def paste_image_with_offset(large_img, small_img, x_offset, y_offset):
  """
  Args:
    large image - image to paste on
    small image - image to paste
    x_offset, y_offset - offsets in pixels for pasting

  Returns:
    Large image with pasted small image on top

  Assumptions:
    Number of channels of images coincide
    Pasted image is small enough to be pasted with required offset
  """
  img_copy = large_img.copy()
  small_h, small_w = small_img.shape[0], small_img.shape[1]
  img_copy[y_offset:y_offset+small_h, x_offset:x_offset+small_w] = small_img
  return img_copy
