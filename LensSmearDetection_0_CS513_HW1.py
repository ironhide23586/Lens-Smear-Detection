import cv2
import glob, os
import numpy as np
import copy
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import Pool, cpu_count
import itertools
import matplotlib.pyplot as plt

def show(img, res=(700, 700)):
  cv2.imshow('image', cv2.resize(img, res))
  cv2.waitKey()

def toBGR(cv2_img):
  b, g, r = [cv2_img[:, :, i] for i in xrange(3)]
  return np.array([b, g, r])

def tocv2(bgr):
  ret = np.array([bgr[:, i, :].T for i in xrange(bgr.shape[1])])
  return ret.astype(np.uint8)

def extract_bgr_channel_set(bgr_sets, channel_idx):
  return bgr_sets[:, channel_idx, :, :]

def get_max_min_maps(mats):
  b = mats.min(axis=0)
  a = mats.max(axis=0)
  return a, b

def get_bgr_max_min_maps(bgr_img_separated):
  maps = np.array([get_max_min_maps(bis) for bis in bgr_img_separated])
  return maps

def get_bgr_max_min(imgs):
  bgr_sets = np.array([toBGR(img) for img in imgs])
  bgr_img_separated = np.array([extract_bgr_channel_set(bgr_sets, c) for c in xrange(imgs.shape[-1])])
  bgr_img_max_min = get_bgr_max_min_maps(bgr_img_separated)
  return bgr_img_max_min

def read_imgs(all_pics, start_idx, end_idx_range):
  if end_idx_range >= all_pics.shape[0]:
    end_idx_range = all_pics.shape[0]
  imgs = np.array([cv2.imread(all_pics[i]) for i in xrange(start_idx, end_idx_range)])
  return imgs

def merge_max_mins(mm0, mm1):
  maxs = np.max([mm0[:, 0, :, :], mm1[:, 0, :, :]], axis=0)
  mins = np.min([mm0[:, 1, :, :], mm1[:, 1, :, :]], axis=0)
  ans = np.array([[maxs[i, :, :], mins[i, :, :]] for i in xrange(maxs.shape[0])])
  return ans

def get_att_scatt(imgs_max_min):
  scatt = imgs_max_min[:, 1, :, :]
  att = (imgs_max_min[:, 0, :, :] - scatt) / 255.
  scatt = scatt / 255.
  return att, scatt

def show_diff(cv2imgs, idx0, idx1):
  show(tocv2(np.diff(np.array([toBGR(cv2img) for cv2img in [cv2imgs[idx0], cv2imgs[idx1]]]), axis=0)[0]))

def img_diff(imgs):
  return np.abs(imgs[0].astype(int) - imgs[1].astype(int)).astype(np.uint8)

def get_common_img(bgr_imgs, thres=5):
  bi = copy.deepcopy(bgr_imgs)
  bi_lst = [(bi[i], bi[i+1]) for i in xrange(bi.shape[0] - 1)]
  p = Pool(cpu_count())
  while bi.shape[0] > 1:
    diffs = np.array(p.map(img_diff, bi_lst))
    #diffs = np.array([img_diff([bi[i], bi[i + 1]]) for i in xrange(bi.shape[0] - 1)])
    diffs[diffs <= thres] = 1
    diffs[diffs > thres] = 0
    bi = (bi[:-1] * diffs).astype(np.uint8)
    bi_lst = [(bi[i], bi[i+1]) for i in xrange(bi.shape[0] - 1)]
    print bi.shape[0]
  p.close()
  return bi[0]

def imgs2bgrimgs(cv2imgs):
  return np.array([toBGR(cv2img) for cv2img in cv2imgs])

def img_grad(bgr_img):
  src = gaussian_blur(bgr_img)
  src_bw = cv2.cvtColor(tocv2(src), cv2.COLOR_BGR2GRAY)
  sobelx = cv2.Sobel(src_bw, cv2.CV_64F, 1, 0, ksize=3)
  sobely = cv2.Sobel(src_bw, cv2.CV_64F, 0, 1, ksize=3)
  grads = np.mean(np.abs([sobelx, sobely]), axis=0)
  return grads

def min_max_normalize(mat):
  max_elem = mat.max()
  min_elem = mat.min()
  m_std = (mat - min_elem) / (max_elem - min_elem)
  m_std *= 255
  m_std = m_std.astype(np.uint8)
  return m_std

def gaussian_blur(bgr_img, kernel_size=3):
  return np.array([gaussian_filter(bgr_img[i], kernel_size) for i in xrange(bgr_img.shape[0])])

def draw_points(img, points, rad=10, color=(128,245,168), thick=-6):
  for point in points:
    cv2.circle(img, point, rad, color, thick)

def draw_and_show(img, point_sets, rad=10, color=(128, 245, 168), thick=-6):
  im = copy.deepcopy(img)
  #colors = [(128, 245, 168), (245, 128, 168), (128, 168, 245)]
  colors = list(itertools.permutations(color))
  i = 0
  for point_set in point_sets:
    draw_points(im, point_set, rad, colors[i], thick)
    i += 1
  show(im)

def img_idx(last3digits):
  return last3digits - 606

def get_u_d(i0, mid):
  b0 = toBGR(i0)
  b0_u = np.mean(b0[:, :mid, :])
  b0_d = np.mean(b0[:, mid:, :])
  return b0_u, b0_d

def absdiff(a):
  a = a.astype(np.int)
  return np.array([np.abs(a[i] - a[i - 1]) for i in xrange(1, a.shape[0])])

def get_max_grad_idx(bgr_col):
  bgr_max_grads = [absdiff(bgr_col_e).argmax() for bgr_col_e in bgr_col]
  return bgr_max_grads

def get_max_grad_idx_single(bgr_col):
  g = get_max_grad_idx(bgr_col)
  return int(np.mean(g))

def get_separator(bgr_img):
  #separator = np.array([get_max_grad_idx_single(bgr_img[:,:,i]) for i in xrange(bgr_img.shape[2])])
  p = Pool(cpu_count())
  separator = p.map(get_max_grad_idx_single, [bgr_img[:, :, i] for i in xrange(bgr_img.shape[2])])
  p.close()
  return np.array(separator)

def draw_sep(im, filt_thres=60):
  b = toBGR(im)
  s = get_separator(b)
  s_grads = absdiff(s)
  tmp = np.zeros(s.shape[0])
  tmp[1:] = s_grads
  s_grads = tmp
  filt = s_grads > filt_thres
  s[filt] = 0
  #r = s.min()
  #pnts = [(i, r) for i in xrange(s.shape[0])]
  pnts = [(i, s[i]) for i in xrange(s.shape[0])]
  draw_and_show(im, [pnts])

def get_sep_row(im):
  b = toBGR(im)
  return get_separator(b).min()

def get_sep_u_d_means(im):
  r = get_sep_row(im)
  return get_u_d(im, r)

def is_desired(im):
  a, b = get_sep_u_d_means(im)
  if a < b:
    return True
  return False

def extract_tunnel_images(read_folder, write_folder, lim=1000, batch_size=100, thres=0):
  all_pics = np.array(glob.glob(read_folder + os.sep + '*'))[400:]
  num_pics = all_pics.shape[0]
  num_batches = int(np.ceil(1. * num_pics / batch_size))
  imgs_max_min = None
  p = Pool(cpu_count())
  masks = []
  for i in xrange(num_batches):
    imgs = read_imgs(all_pics, i * batch_size, (i + 1) * batch_size)
    names_paths = np.array(all_pics[i * batch_size: (i + 1) * batch_size])
    names = np.array([name.split(os.sep)[-1] for name in names_paths])
    sel_filter = []
    i = 0
    for img in imgs:
      print '\nProcessing image', names[i]
      if is_desired(img) == True:
        print '->Accepted'
        cv2.imwrite(fld + os.sep + names[i], img)
      else:
        print '->Rejected HUEHUEHUEHUE'
      i += 1
  p.close()

def get_center(point_set_inp):
  point_set = np.array(point_set_inp)
  a = np.mean(point_set, axis=0).astype(np.int)
  return (a[0], a[1])

def extract_bgr_pixel_seq(bgr_imgs, point):
  return bgr_imgs[:, :, point[0], point[1]]

def get_stack_pixels_bgr(imgs, x, y):
  r = y
  c = x
  b = imgs[:, 0, r, c]
  g = imgs[:, 1, r, c]
  r = imgs[:, 2, r, c]
  return b, g, r

def process_folder(read_folder, write_folder, lim=1000, batch_size=100, thres=0):
  all_pics = np.array(glob.glob(read_folder + os.sep + '*'))[:lim]
  num_pics = all_pics.shape[0]
  num_batches = int(np.ceil(1. * num_pics / batch_size))
  imgs_max_min = None
  p = Pool(cpu_count())
  masks = []
  average_img=np.zeros((2032, 2032, 3),np.float)
  smear_point_xy = (948, 717)
  non_smear_point_xy = (1017, 708)
  for i in xrange(num_batches):
    imgs = read_imgs(all_pics, i * batch_size, (i + 1) * batch_size)
    bi = imgs2bgrimgs(imgs)
    k=0

def get_smear_mask_sobel(read_folder, write_folder, lim=1000, batch_size=100, thres=0):
  all_pics = np.array(glob.glob(read_folder + os.sep + '*'))[:lim]
  num_pics = all_pics.shape[0]
  num_batches = int(np.ceil(1. * num_pics / batch_size))
  imgs_max_min = None
  p = Pool(cpu_count())
  masks = []
  average_img=np.zeros((2032, 2032, 3),np.float)
  for i in xrange(num_batches):
    imgs = read_imgs(all_pics, i * batch_size, (i + 1) * batch_size)
    bi = imgs2bgrimgs(imgs)
    mask = np.sum(p.map(img_grad, bi), axis=0)
    mask /= batch_size * 1.
    mask_mask = np.ones(mask.shape)
    mask_mask[:, 0] = 0
    mask_mask[:, -1] = 0
    mask_mask[0, :] = 0
    mask_mask[-1, :] = 0
    mask_mask *= mask
    mask = min_max_normalize(mask_mask)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=4)
    thres = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY_INV)
    masks.append(mask)
    if (i + 1) * batch_size < num_pics:
      print 'Processed', (i + 1) * batch_size, 'images'
    else:
      print 'Processed', num_pics, 'images'
  masks = np.array(masks)
  m = np.mean(masks, axis=0)
  m2 = min_max_normalize(m)
  cv2.imwrite(fld + os.sep + '_'.join(['mask', str(lim) + 'i', str(batch_size) + 'bs', '_'.join(read_folder.split(os.sep))]) + '.jpg', m2)

def get_mask(folder_path, write_folder, lim=1000, batch_size=100, thres=0):
  write_folder += '_att_scatt_maps'
  if not os.path.isdir(write_folder):
    os.makedirs(write_folder)
  all_pics = np.array(glob.glob(folder_path + os.sep + '*'))[:lim]
  num_pics = all_pics.shape[0]
  num_batches = int(np.ceil(1. * num_pics / batch_size))
  imgs_max_min = None
  for i in xrange(num_batches):
    imgs = read_imgs(all_pics, i * batch_size, (i + 1) * batch_size)
    imgs_max_min_new = get_bgr_max_min(imgs)
    if i == 0:
      imgs_max_min = copy.deepcopy(imgs_max_min_new)
    else:
      imgs_max_min = merge_max_mins(imgs_max_min, imgs_max_min_new)
    if (i + 1) * batch_size < num_pics:
      print 'Processed', (i + 1) * batch_size, 'images'
    else:
      print 'Processed', num_pics, 'images'
  attenuations, scatters = get_att_scatt(imgs_max_min)
  attenuations = (attenuations * 255).astype(np.uint8)
  scatters = (scatters * 255).astype(np.uint8)
  attenuations_imgs = tocv2(attenuations)
  scatter_imgs = tocv2(scatters)
  cv2.imwrite(write_folder + os.sep + 'att_' + str(lim) + 'i.jpg', attenuations_imgs)
  cv2.imwrite(write_folder + os.sep + 'scatt_' + str(lim) + 'i.jpg', scatter_imgs)
  net_effect = (attenuations_imgs * (scatter_imgs / 255.)).astype(np.uint8)
  thres = net_effect.mean()
  cv2.imwrite(write_folder + os.sep + 'att_mul_scatt_' + str(lim) + 'i_' + str(thres) + 'thres.jpg', net_effect)
  cv2.imwrite(write_folder + os.sep + 'att_mul_scatt_' + str(lim) + 'i_' + str(thres) + 'thres_b.jpg', net_effect[:, :, 0])
  cv2.imwrite(write_folder + os.sep + 'att_mul_scatt_' + str(lim) + 'i_' + str(thres) + 'thres_g.jpg', net_effect[:, :, 1])
  cv2.imwrite(write_folder + os.sep + 'att_mul_scatt_' + str(lim) + 'i_' + str(thres) + 'thres_r.jpg', net_effect[:, :, 2])
  k=0

def get_conv(img, kernel_size, showImg=True):
  #kernel = np.array([[-1, -1, -1], 
  #                   [-1, 9, -1],
  #                   [-1, -1, -1]])
  kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
  im = cv2.filter2D(img, -1, kernel)
  if showImg == True:
    show(im)
  return im

def imhist(img_bw, show_plot=True):
  hist, bins = np.histogram(img_bw.ravel(), 20, [0, 256])
  bins = bins[:-1].astype(np.int)
  if show_plot == True:
    ind = np.arange(1, bins.shape[0] + 1)  # the x locations for the groups
    maxs = hist
    width = 0.35       # the width of the bars
    fig, ax = plt.subplots()
    rects_mid = ax.bar(ind, maxs, .2, color='r') #Use for Accuracy
    ax.set_ylabel('Count')
    ax.set_xlabel('Pixel Value')
    ax.set_xticks(ind + .5*.2) #Use for Accuracy
    ax.set_xticklabels(bins)
    plt.show()
  return hist, bins

def show_thres(img, upper_lim):
  im = img
  im[img > upper_lim] = 0
  show(im)

def arr_ratios(arr):
  return arr[1:] / arr[:-1]

def get_mask_threshold(counts, pixels):
  r = arr_ratios(counts)
  indices = r.argsort()[-2:] + 1
  if pixels[indices[0]] > pixels[indices[1]]:
    return pixels[indices[0]]
  return pixels[indices[1]]

def process_masks(folder_path):
  img_paths = np.array(glob.glob(folder_path + os.sep + '*.jpg'))
  img_paths = np.array([ip for ip in img_paths if 'bw' not in ip])
  img_paths = np.array([ip for ip in img_paths if 'conv' not in ip])
  imgs = read_imgs(img_paths, 0, img_paths.shape[0])
  kernel_size = 60
  kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
  imgs_bw = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs])
  imgs_conv = np.array([cv2.filter2D(img, -1, kernel) for img in imgs])
  i = 0
  for i in xrange(imgs_bw.shape[0]):
    base_name = '.'.join(img_paths[i].split(os.sep)[-1].split('.')[:-1]) 
    bw_name = base_name + '_bw.jpg'
    conv_name = base_name + '_conv' + str(kernel_size) + 'kernel.jpg'
    bw_hist_name = base_name + '_bw_hist.jpg'
    img_bw = imgs_bw[i]
    counts, pixels = imhist(img_bw, False)
    thres = get_mask_threshold(counts, pixels)

    img_bw[img_bw > thres] = 0

    show(img_bw)
    #cv2.imwrite(folder_path + os.sep + bw_name, imgs_bw[i])
    #cv2.imwrite(folder_path + os.sep + conv_name, imgs_conv[i])
    i += 1
  k=0

#this method is garbage
def show_conn_areas(img, area_min, area_max):
  o = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
  stats = o[2]
  cen = o[3]
  f0 = stats[:,-1] < area_max
  f1 = stats[:,-1] > area_min
  f = f0 & f1
  #aoi = stats[f]
  #cen = cen[f]
  filt_labels = np.arange(1, o[0] + 1)[f]
  im = o[2]
  for filt_label in filt_labels:
    im[im==filt_label] = -1
  im[im>-1] = 0
  im[im==-1] = 255
  im = im.astype(np.uint8)
  show(im)

def bw2bgr(bw_im):
  m = np.zeros([3, bw_im.shape[0], bw_im.shape[1]])
  for i in xrange(3):
    m[i] = bw_im
  return m.astype(np.uint8)

def get_regions(bw_im, min_area, max_area, show_img=False):
  im, contours, h = cv2.findContours(bw_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  contours = np.array(contours)
  areas = np.array([cv2.contourArea(cnt) for cnt in contours])
  perims = np.array([cv2.arcLength(cnt, True) for cnt in contours])
  f0 = areas < max_area
  f1 = areas > min_area
  filt = f0 & f1
  contours_filt = contours[filt]
  a = np.array([cv2.contourArea(cnt) for cnt in contours_filt])
  p = np.array([cv2.arcLength(cnt, True) for cnt in contours_filt])
  bgr_bw_img = bw2bgr(im)
  ans = cv2.drawContours(tocv2(bgr_bw_img), contours_filt, -1, (0,255,0), 3)
  if show_img == True:
    show(ans)
  return ans

def get_grayscale_mask(folder_path):
  cam_idx = int(folder_path.split('_')[1][3:])
  img_paths = np.array(glob.glob(folder_path + os.sep + '*.jpg'))
  img_paths = np.array([ip for ip in img_paths if 'bw' not in ip])
  img_paths = np.array([ip for ip in img_paths if 'conv' not in ip])
  imgs = read_imgs(img_paths[:1], 0, img_paths.shape[0])
  img_bw = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
  base_name = '.'.join(img_paths[0].split(os.sep)[-1].split('.')[:-1]) 
  counts, pixels = imhist(img_bw, False)
  thres = get_mask_threshold(counts, pixels)
  img_bw[img_bw > thres] = 0
  i = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, np.ones([3, 3]))
  k = cv2.morphologyEx(i, cv2.MORPH_CLOSE, np.ones([8, 8]))
  k = cv2.threshold(k, 0, 255, cv2.THRESH_BINARY)[1]
  if cam_idx == 1:
    img_bw = get_regions(k, 14000, 20000) #cam1
  elif cam_idx == 2:
    img_bw = get_regions(k, 1600, 2000) #cam2
  else:
    img_bw = get_regions(k, 10000, 16000) #cam3
  mask_name = base_name + '_mask_highlighted.jpg'
  cv2.imwrite(folder_path + os.sep + mask_name, img_bw)
  #return img_bw

if __name__ == "__main__":
  lim = 4500
  cam_idx = 3
  root_folder = 'sample_drive'


  img_folder = 'cam_'+ str(cam_idx)
  pic_folders = glob.glob(root_folder + os.sep + img_folder)
  fld = '_'.join([str(lim) + 'i', 'cam' + str(cam_idx)])
  #process_folder(pic_folders[0], fld, lim, 50)
  
  extract_tunnel_images(pic_folders[0], 'filt_imgs', lim, 50)
  get_mask(pic_folders[0], fld, lim, 50)
  get_grayscale_mask(fld + '_att_scatt_maps')
  #process_masks('4500i_cam' + str(cam_idx) + '_att_scatt_maps')