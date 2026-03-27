import cv2
import numpy as np

def add_alpha_channel(img):
    """ 为jpg图像添加alpha通道 """

    b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道

    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
    return img_new

# def paste_bbox_img(jpg_img, png_img, y1, y2, x1, x2):
def paste_bbox_img(img1, iou, template, alpha):
    """ 将png透明图像与jpg图像叠加
        y1,y2,x1,x2为叠加位置坐标值
    """
    # 判断jpg图像是否已经为4通道
    if img1.shape[2] == 3:
        img1 = add_alpha_channel(img1)
    print(img1.shape)

    x1 = iou[0]
    x2 = iou[2]
    y1 = iou[1]
    y2 = iou[3]

    yy1 = 0
    yy2 = template.shape[0]
    xx1 = 0
    xx2 = template.shape[1]

    # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
    alpha_png = template[yy1:yy2, xx1:xx2, 3] / 255.0 * alpha
    alpha_jpg = 1 - alpha_png

    print(x1, x2, y1, y2, yy1, yy2, xx1, xx2)

    # 开始叠加
    for c in range(0, 3):
        img1[y1:y2, x1:x2, c] = ((alpha_jpg * img1[y1:y2, x1:x2, c]) + (alpha_png * template[yy1:yy2, xx1:xx2, c]))

    return img1

# ------------ test_main -------------
resize_H = 1080
resize_W = 1920
part_W = round(resize_W / 16)
part_W_l = round(resize_W / 16 * 1.2)
part_H = round(resize_H / 8)

main_icon_small_bg = cv2.imread("E:/gui_test/main/1.png", cv2.IMREAD_UNCHANGED)
main_icon_large_left_bg = cv2.imread('E:/gui_test/main/2_left.png', cv2.IMREAD_UNCHANGED)
main_icon_large_right_bg = cv2.imread('E:/gui_test/main/2_right.png', cv2.IMREAD_UNCHANGED)
main_icon_small_bg = cv2.resize(main_icon_small_bg, (part_W_l, part_H * 2))
main_icon_large_left_bg = cv2.resize(main_icon_large_left_bg, (part_W * 2, part_H * 2))
main_icon_large_right_bg = cv2.resize(main_icon_large_right_bg, (part_W * 2, part_H * 2))

main_uterus_icon_small = cv2.imread("E:/gui_test/main/uterus_1.png", cv2.IMREAD_UNCHANGED)
main_uterus_left_icon_large = cv2.imread("E:/gui_test/main/uterus_2_left.png", cv2.IMREAD_UNCHANGED)
main_uterus_right_icon_large = cv2.imread("E:/gui_test/main/uterus_2_right.png", cv2.IMREAD_UNCHANGED)
main_uterus_icon_small = cv2.resize(main_uterus_icon_small, (part_W_l, part_H * 2))
main_uterus_left_icon_large = cv2.resize(main_uterus_left_icon_large, (part_W * 2, part_H * 2))
main_uterus_right_icon_large = cv2.resize(main_uterus_right_icon_large, (part_W * 2, part_H * 2))

main_domain_icon_small = cv2.imread('E:/gui_test/main/main_1.png', cv2.IMREAD_UNCHANGED)
main_domain_left_icon_large = cv2.imread('E:/gui_test/main/main_2_left.png', cv2.IMREAD_UNCHANGED)
main_domain_right_icon_large = cv2.imread('E:/gui_test/main/main_2_right.png', cv2.IMREAD_UNCHANGED)
main_domain_icon_small = cv2.resize(main_domain_icon_small, (part_W_l, part_H * 2))
main_domain_left_icon_large = cv2.resize(main_domain_left_icon_large, (part_W * 2, part_H * 2))
main_domain_right_icon_large = cv2.resize(main_domain_right_icon_large, (part_W * 2, part_H * 2))

main_switch_icon_small = cv2.imread("E:/gui_test/main/switch_1.png", cv2.IMREAD_UNCHANGED)
main_switch_left_icon_large = cv2.imread("E:/gui_test/main/switch_2_left.png", cv2.IMREAD_UNCHANGED)
main_switch_right_icon_large = cv2.imread("E:/gui_test/main/switch_2_right.png", cv2.IMREAD_UNCHANGED)
main_switch_icon_small = cv2.resize(main_switch_icon_small, (part_W_l, part_H*2))
main_switch_left_icon_large = cv2.resize(main_switch_left_icon_large, (part_W*2, part_H*2))
main_switch_right_icon_large = cv2.resize(main_switch_right_icon_large, (part_W*2, part_H*2))

main_camera_icon_small = cv2.imread("E:/gui_test/main/camera_1.png", cv2.IMREAD_UNCHANGED)
main_camera_left_icon_large = cv2.imread("E:/gui_test/main/camera_2_left.png", cv2.IMREAD_UNCHANGED)
main_camera_right_icon_large = cv2.imread("E:/gui_test/main/camera_2_right.png", cv2.IMREAD_UNCHANGED)
main_camera_icon_small = cv2.resize(main_camera_icon_small, (part_W_l, part_H*2))
main_camera_left_icon_large = cv2.resize(main_camera_left_icon_large, (part_W*2, part_H*2))
main_camera_right_icon_large = cv2.resize(main_camera_right_icon_large, (part_W*2, part_H*2))

# Main panel small bbox left
main_uterus_bbox_left_small = [0, 0, part_W_l, 2*part_H] # x1, y1, x2, y2
main_dominant_bbox_left_small = [0, 2 * part_H, part_W_l, 4 * part_H]  # x1, y1, x2, y2
main_switch_bbox_left_small = [0, 4*part_H, part_W_l, 6*part_H] # x1, y1, x2, y2
main_camera_bbox_left_small = [0, 6*part_H, part_W_l, 8*part_H] # x1, y1, x2, y2
# large bbox left
main_uterus_bbox_left_large = [0, 0, 2 * part_W, 2 * part_H]  # x1, y1, x2, y2
main_dominant_bbox_left_large = [0, 2 * part_H, 2 * part_W, 4 * part_H]  # x1, y1, x2, y2
main_switch_bbox_left_large = [0, 4 * part_H, 2 * part_W, 6 * part_H]  # x1, y1, x2, y2
main_camera_bbox_left_large = [0, 6 * part_H, 2 * part_W, 8 * part_H]  # x1, y1, x2, y2
# small bbox right
main_uterus_bbox_right_small = [resize_W - part_W_l, 0, resize_W, 2 * part_H]  # x1, y1, x2, y2
main_dominant_bbox_right_small = [resize_W - part_W_l, 2 * part_H, resize_W, 4 * part_H]  # x1, y1, x2, y2
main_switch_bbox_right_small = [resize_W - part_W_l, 4 * part_H, resize_W, 6 * part_H]  # x1, y1, x2, y2
main_camera_bbox_right_small = [resize_W - part_W_l, 6 * part_H, resize_W, 8 * part_H]  # x1, y1, x2, y2
# large bbox right
main_uterus_bbox_right_large = [resize_W - 2 * part_W, 0, resize_W, 2 * part_H]  # x1, y1, x2, y2
main_dominant_bbox_right_large = [resize_W - 2 * part_W, 2 * part_H, resize_W, 4 * part_H]  # x1, y1, x2, y2
main_switch_bbox_right_large = [resize_W - 2 * part_W, 4 * part_H, resize_W, 6 * part_H]  # x1, y1, x2, y2
main_camera_bbox_right_large = [resize_W - 2 * part_W, 6 * part_H, resize_W, 8 * part_H]  # x1, y1, x2, y2

# ------------ test_camera -----------
camera_panel_W = round(resize_W / 8)
camera_panel_H = round(3 * resize_H / 4 / 4)

camera_panel_large_left_bg = cv2.imread('E:/gui_test/camera/3_left.png', cv2.IMREAD_UNCHANGED)
camera_panel_large_right_bg = cv2.imread('E:/gui_test/camera/3_right.png', cv2.IMREAD_UNCHANGED)
camera_panel_small_left_bg = cv2.imread('E:/gui_test/camera/4_left.png', cv2.IMREAD_UNCHANGED)
camera_panel_small_right_bg = cv2.imread('E:/gui_test/camera/4_right.png', cv2.IMREAD_UNCHANGED)
camera_panel_small_zero_left_bg = cv2.imread('E:/gui_test/camera/4_left.png', cv2.IMREAD_UNCHANGED)
camera_panel_small_zero_right_bg = cv2.imread('E:/gui_test/camera/4_right.png', cv2.IMREAD_UNCHANGED)
camera_panel_large_left_bg = cv2.resize(camera_panel_large_left_bg, (part_W * 2, part_H * 2))
camera_panel_large_right_bg = cv2.resize(camera_panel_large_right_bg, (part_W * 2, part_H * 2))
camera_panel_small_left_bg = cv2.resize(camera_panel_small_left_bg, (camera_panel_W, camera_panel_H))
camera_panel_small_right_bg = cv2.resize(camera_panel_small_right_bg, (camera_panel_W, camera_panel_H))
camera_panel_small_zero_left_bg = cv2.resize(camera_panel_small_zero_left_bg, (camera_panel_W, camera_panel_H+2))
camera_panel_small_zero_right_bg = cv2.resize(camera_panel_small_zero_right_bg, (camera_panel_W, camera_panel_H+2))

# ------------ camera left -----------
camera_panel_click_icon_large_left = cv2.imread('E:/gui_test/camera/camera_3_left.png', cv2.IMREAD_UNCHANGED)
camera_panel_click_icon_large_left = cv2.resize(camera_panel_click_icon_large_left, (part_W * 2, part_H * 2))

camera_panel_zoomin_icon_left = cv2.imread('E:/gui_test/camera/in_left.png', cv2.IMREAD_UNCHANGED)
camera_panel_zoomout_icon_left = cv2.imread('E:/gui_test/camera/out_left.png', cv2.IMREAD_UNCHANGED)
camera_panel_zoomauto_icon_left = cv2.imread('E:/gui_test/camera/track_left.png', cv2.IMREAD_UNCHANGED)
camera_panel_setzero_icon_left = cv2.imread('E:/gui_test/camera/zero_left.png', cv2.IMREAD_UNCHANGED)
camera_panel_zoomin_icon_left = cv2.resize(camera_panel_zoomin_icon_left, (camera_panel_W, camera_panel_H))
camera_panel_zoomout_icon_left = cv2.resize(camera_panel_zoomout_icon_left, (camera_panel_W, camera_panel_H))
camera_panel_zoomauto_icon_left = cv2.resize(camera_panel_zoomauto_icon_left, (camera_panel_W, camera_panel_H))
camera_panel_setzero_icon_left = cv2.resize(camera_panel_setzero_icon_left, (camera_panel_W, camera_panel_H+2))

camera_panel_zoomin_click_icon_left = cv2.imread('E:/gui_test/camera/in_click_left.png', cv2.IMREAD_UNCHANGED)
camera_panel_zoomout_click_icon_left = cv2.imread('E:/gui_test/camera/out_click_left.png', cv2.IMREAD_UNCHANGED)
camera_panel_zoomauto_click_icon_left = cv2.imread('E:/gui_test/camera/track_click_left.png', cv2.IMREAD_UNCHANGED)
camera_panel_setzero_click_icon_left = cv2.imread('E:/gui_test/camera/zero_click_left.png', cv2.IMREAD_UNCHANGED)
camera_panel_zoomin_click_icon_left = cv2.resize(camera_panel_zoomin_click_icon_left, (camera_panel_W, camera_panel_H))
camera_panel_zoomout_click_icon_left = cv2.resize(camera_panel_zoomout_click_icon_left, (camera_panel_W, camera_panel_H))
camera_panel_zoomauto_click_icon_left = cv2.resize(camera_panel_zoomauto_click_icon_left, (camera_panel_W, camera_panel_H))
camera_panel_setzero_click_icon_left = cv2.resize(camera_panel_setzero_click_icon_left, (camera_panel_W, camera_panel_H+2))

# ------------ camera right -----------
camera_panel_click_icon_large_right = cv2.imread('E:/gui_test/camera/camera_3_right.png', cv2.IMREAD_UNCHANGED)
camera_panel_click_icon_large_right = cv2.resize(camera_panel_click_icon_large_right, (part_W * 2, part_H * 2))

camera_panel_zoomin_icon_right = cv2.imread('E:/gui_test/camera/in_right.png', cv2.IMREAD_UNCHANGED)
camera_panel_zoomout_icon_right = cv2.imread('E:/gui_test/camera/out_right.png', cv2.IMREAD_UNCHANGED)
camera_panel_zoomauto_icon_right = cv2.imread('E:/gui_test/camera/track_right.png', cv2.IMREAD_UNCHANGED)
camera_panel_setzero_icon_right = cv2.imread('E:/gui_test/camera/zero_right.png', cv2.IMREAD_UNCHANGED)
camera_panel_zoomin_icon_right = cv2.resize(camera_panel_zoomin_icon_right, (camera_panel_W, camera_panel_H))
camera_panel_zoomout_icon_right = cv2.resize(camera_panel_zoomout_icon_right, (camera_panel_W, camera_panel_H))
camera_panel_zoomauto_icon_right = cv2.resize(camera_panel_zoomauto_icon_right, (camera_panel_W, camera_panel_H))
camera_panel_setzero_icon_right = cv2.resize(camera_panel_setzero_icon_right, (camera_panel_W, camera_panel_H+2))

camera_panel_zoomin_click_icon_right = cv2.imread('E:/gui_test/camera/in_click_right.png', cv2.IMREAD_UNCHANGED)
camera_panel_zoomout_click_icon_right = cv2.imread('E:/gui_test/camera/out_click_right.png', cv2.IMREAD_UNCHANGED)
camera_panel_zoomauto_click_icon_right = cv2.imread('E:/gui_test/camera/track_click_right.png', cv2.IMREAD_UNCHANGED)
camera_panel_setzero_click_icon_right = cv2.imread('E:/gui_test/camera/zero_click_right.png', cv2.IMREAD_UNCHANGED)
camera_panel_zoomin_click_icon_right = cv2.resize(camera_panel_zoomin_click_icon_right, (camera_panel_W, camera_panel_H))
camera_panel_zoomout_click_icon_right = cv2.resize(camera_panel_zoomout_click_icon_right, (camera_panel_W, camera_panel_H))
camera_panel_zoomauto_click_icon_right = cv2.resize(camera_panel_zoomauto_click_icon_right, (camera_panel_W, camera_panel_H))
camera_panel_setzero_click_icon_right = cv2.resize(camera_panel_setzero_click_icon_right, (camera_panel_W, camera_panel_H+2))

camera_panel_zoomin_bbox_left = [0, 0, camera_panel_W, camera_panel_H]  # x1, y1, x2, y2
camera_panel_zoomout_bbox_left = [0, camera_panel_H, camera_panel_W, 2 * camera_panel_H]  # x1, y1, x2, y2
camera_panel_zoomauto_bbox_left = [0, 2 * camera_panel_H, camera_panel_W, 3 * camera_panel_H]  # x1, y1, x2, y2
camera_panel_setzero_bbox_left = [0, 3 * camera_panel_H, camera_panel_W, 4 * camera_panel_H+2]  # x1, y1, x2, y2
#
camera_panel_zoomin_bbox_right = [resize_W - camera_panel_W, 0, resize_W, camera_panel_H]  # x1, y1, x2, y2
camera_panel_zoomout_bbox_right = [resize_W - camera_panel_W, camera_panel_H, resize_W,
                                        2 * camera_panel_H]  # x1, y1, x2, y2
camera_panel_zoomauto_bbox_right = [resize_W - camera_panel_W, 2 * camera_panel_H, resize_W,
                                         3 * camera_panel_H]  # x1, y1, x2, y2
camera_panel_setzero_bbox_right = [resize_W - camera_panel_W, 3 * camera_panel_H, resize_W,
                                        4 * camera_panel_H+2]  # x1, y1, x2, y2

# ------------ test_uterus ----------
uterus_panel_W = round(resize_W / 16)
uterus_panel_H = round(resize_H / 8)

uterus_panel_large_left_bg = cv2.imread('E:/gui_test/uterus/3_left.png', cv2.IMREAD_UNCHANGED)
uterus_panel_large_right_bg = cv2.imread('E:/gui_test/uterus/3_right.png', cv2.IMREAD_UNCHANGED)
uterus_panel_up_bg = cv2.imread('E:/gui_test/uterus/up_bg.png', cv2.IMREAD_UNCHANGED)
uterus_panel_down_bg = cv2.imread('E:/gui_test/uterus/down_bg.png', cv2.IMREAD_UNCHANGED)
uterus_panel_left_bg = cv2.imread('E:/gui_test/uterus/left_bg.png', cv2.IMREAD_UNCHANGED)
uterus_panel_right_bg = cv2.imread('E:/gui_test/uterus/right_bg.png', cv2.IMREAD_UNCHANGED)
uterus_panel_insert_bg = cv2.imread('E:/gui_test/uterus/insert_bg.png', cv2.IMREAD_UNCHANGED)
uterus_panel_retract_bg = cv2.imread('E:/gui_test/uterus/retract_bg.png', cv2.IMREAD_UNCHANGED)
uterus_panel_large_left_bg = cv2.resize(uterus_panel_large_left_bg, (part_W * 2, part_H * 2))
uterus_panel_large_right_bg = cv2.resize(uterus_panel_large_right_bg, (part_W * 2, part_H * 2))
uterus_panel_up_bg = cv2.resize(uterus_panel_up_bg, (uterus_panel_W * 4, uterus_panel_H))
uterus_panel_down_bg = cv2.resize(uterus_panel_down_bg, (uterus_panel_W * 4, uterus_panel_H))
uterus_panel_left_bg = cv2.resize(uterus_panel_left_bg, (part_W * 2, part_H * 2))
uterus_panel_right_bg = cv2.resize(uterus_panel_right_bg, (part_W * 2, part_H * 2))
uterus_panel_insert_bg = cv2.resize(uterus_panel_insert_bg, (part_W * 2, part_H * 2))
uterus_panel_retract_bg = cv2.resize(uterus_panel_retract_bg, (part_W * 2, part_H * 2))

uterus_panel_click_icon_large_left = cv2.imread('E:/gui_test/uterus/uterus_3_left.png', cv2.IMREAD_UNCHANGED)
uterus_panel_click_icon_large_left = cv2.resize(uterus_panel_click_icon_large_left, (part_W * 2, part_H * 2))
uterus_panel_click_icon_large_right = cv2.imread('E:/gui_test/uterus/uterus_3_right.png', cv2.IMREAD_UNCHANGED)
uterus_panel_click_icon_large_right = cv2.resize(uterus_panel_click_icon_large_right, (part_W * 2, part_H * 2))

uterus_panel_down_icon = cv2.imread('E:/gui_test/uterus/down.png', cv2.IMREAD_UNCHANGED)
uterus_panel_up_icon = cv2.imread('E:/gui_test/uterus/up.png', cv2.IMREAD_UNCHANGED)
uterus_panel_left_icon = cv2.imread('E:/gui_test/uterus/left.png', cv2.IMREAD_UNCHANGED)
uterus_panel_right_icon = cv2.imread('E:/gui_test/uterus/right.png', cv2.IMREAD_UNCHANGED)
uterus_panel_insert_icon = cv2.imread('E:/gui_test/uterus/insert.png', cv2.IMREAD_UNCHANGED)
uterus_panel_retract_icon = cv2.imread('E:/gui_test/uterus/retract.png', cv2.IMREAD_UNCHANGED)
uterus_panel_down_icon = cv2.resize(uterus_panel_down_icon, (uterus_panel_W * 4, uterus_panel_H))
uterus_panel_up_icon = cv2.resize(uterus_panel_up_icon, (uterus_panel_W * 4, uterus_panel_H))
uterus_panel_left_icon = cv2.resize(uterus_panel_left_icon, (uterus_panel_W * 2, uterus_panel_H * 2))
uterus_panel_right_icon = cv2.resize(uterus_panel_right_icon, (uterus_panel_W * 2, uterus_panel_H * 2))
uterus_panel_insert_icon = cv2.resize(uterus_panel_insert_icon, (uterus_panel_W * 2, uterus_panel_H * 2))
uterus_panel_retract_icon = cv2.resize(uterus_panel_retract_icon, (uterus_panel_W * 2, uterus_panel_H * 2))

uterus_panel_down_click_icon = cv2.imread('E:/gui_test/uterus/down_click.png', cv2.IMREAD_UNCHANGED)
uterus_panel_up_click_icon = cv2.imread('E:/gui_test/uterus/up_click.png', cv2.IMREAD_UNCHANGED)
uterus_panel_left_click_icon = cv2.imread('E:/gui_test/uterus/left_click.png', cv2.IMREAD_UNCHANGED)
uterus_panel_right_click_icon = cv2.imread('E:/gui_test/uterus/right_click.png', cv2.IMREAD_UNCHANGED)
uterus_panel_insert_click_icon = cv2.imread('E:/gui_test/uterus/insert_click.png', cv2.IMREAD_UNCHANGED)
uterus_panel_retract_click_icon = cv2.imread('E:/gui_test/uterus/retract_click.png', cv2.IMREAD_UNCHANGED)
uterus_panel_down_click_icon = cv2.resize(uterus_panel_down_click_icon, (uterus_panel_W * 4, uterus_panel_H))
uterus_panel_up_click_icon = cv2.resize(uterus_panel_up_click_icon, (uterus_panel_W * 4, uterus_panel_H))
uterus_panel_left_click_icon = cv2.resize(uterus_panel_left_click_icon, (uterus_panel_W * 2, uterus_panel_H * 2))
uterus_panel_right_click_icon = cv2.resize(uterus_panel_right_click_icon, (uterus_panel_W * 2, uterus_panel_H * 2))
uterus_panel_insert_click_icon = cv2.resize(uterus_panel_insert_click_icon,
                                                 (uterus_panel_W * 2, uterus_panel_H * 2))
uterus_panel_retract_click_icon = cv2.resize(uterus_panel_retract_click_icon,
                                                  (uterus_panel_W * 2, uterus_panel_H * 2))

uterus_panel_up_bbox = [round(resize_W / 2 - uterus_panel_W * 2), 0, round(resize_W / 2 + uterus_panel_W * 2),
                             uterus_panel_H]  # x1, y1, x2, y2
uterus_panel_down_bbox = [round(resize_W / 2 - uterus_panel_W * 2), resize_H - uterus_panel_H,
                               round(resize_W / 2 + uterus_panel_W * 2), resize_H]  # x1, y1, x2, y2
uterus_panel_left_bbox = [0, round(resize_H / 2 - uterus_panel_H), uterus_panel_W * 2,
                               round(resize_H / 2 + uterus_panel_H)]  # x1, y1, x2, y2
uterus_panel_right_bbox = [round(resize_W - uterus_panel_W * 2), round(resize_H / 2 - uterus_panel_H), resize_W,
                                round(resize_H / 2 + uterus_panel_H)]  # x1, y1, x2, y2
uterus_panel_insert_bbox = main_camera_bbox_left_large  # x1, y1, x2, y2
uterus_panel_retract_bbox = main_camera_bbox_right_large  # x1, y1, x2, y2


# ------------ display -------------
def display_main():
    # img_path = "E:/gui_test/030.jpg"
    img_path = "E:/gui_test/001_000013.jpg"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (resize_W, resize_H))
    # iou = [0, 240, 0, 270]
    # iou = [0, test_w, 0, test_h]
    print(img.shape)
    print(main_icon_small_bg.shape)
    print(main_uterus_icon_small.shape)

    display_main_left_small = True
    display_main_right_small = False
    display_main_left_large = False
    display_main_right_large = True

    # ------------ display main left small -------------
    if display_main_left_small:
        img_results = paste_bbox_img(img, main_uterus_bbox_left_small, main_icon_small_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_uterus_bbox_left_small, main_uterus_icon_small, alpha=0.9)
        img_results = paste_bbox_img(img_results, main_dominant_bbox_left_small, main_icon_small_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_dominant_bbox_left_small, main_domain_icon_small, alpha=0.9)
        img_results = paste_bbox_img(img_results, main_switch_bbox_left_small, main_icon_small_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_switch_bbox_left_small, main_switch_icon_small, alpha=0.9)
        img_results = paste_bbox_img(img_results, main_camera_bbox_left_small, main_icon_small_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_camera_bbox_left_small, main_camera_icon_small, alpha=0.9)

        img_results = cv2.resize(img_results, (960, 540))
        cv2.imshow('display_main_left_small', img_results)

    # ------------ display main right small -------------
    if display_main_right_small:
        img_results = paste_bbox_img(img, main_uterus_bbox_right_small, main_icon_small_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_uterus_bbox_right_small, main_uterus_icon_small, alpha=0.9)
        img_results = paste_bbox_img(img_results, main_dominant_bbox_right_small, main_icon_small_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_dominant_bbox_right_small, main_domain_icon_small, alpha=0.9)
        img_results = paste_bbox_img(img_results, main_switch_bbox_right_small, main_icon_small_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_switch_bbox_right_small, main_switch_icon_small, alpha=0.9)
        img_results = paste_bbox_img(img_results, main_camera_bbox_right_small, main_icon_small_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_camera_bbox_right_small, main_camera_icon_small, alpha=0.9)

        img_results = cv2.resize(img_results, (960, 540))
        cv2.imshow('display_main_right_small', img_results)

    # ------------ display main left large -------------
    if display_main_left_large:
        img_results = paste_bbox_img(img, main_uterus_bbox_left_large, main_icon_large_left_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_uterus_bbox_left_large, main_uterus_left_icon_large, alpha=0.9)
        img_results = paste_bbox_img(img_results, main_dominant_bbox_left_large, main_icon_large_left_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_dominant_bbox_left_large, main_domain_left_icon_large, alpha=0.9)
        img_results = paste_bbox_img(img_results, main_switch_bbox_left_large, main_icon_large_left_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_switch_bbox_left_large, main_switch_left_icon_large, alpha=0.9)
        img_results = paste_bbox_img(img_results, main_camera_bbox_left_large, main_icon_large_left_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_camera_bbox_left_large, main_camera_left_icon_large, alpha=0.9)

        img_results = cv2.resize(img_results, (960, 540))
        cv2.imshow('display_main_left_large', img_results)

    # ------------ display main right large -------------
    if display_main_right_large:
        img_results = paste_bbox_img(img, main_uterus_bbox_right_large, main_icon_large_right_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_uterus_bbox_right_large, main_uterus_right_icon_large, alpha=0.9)
        img_results = paste_bbox_img(img_results, main_dominant_bbox_right_large, main_icon_large_right_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_dominant_bbox_right_large, main_domain_right_icon_large, alpha=0.9)
        img_results = paste_bbox_img(img_results, main_switch_bbox_right_large, main_icon_large_right_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_switch_bbox_right_large, main_switch_right_icon_large, alpha=0.9)
        img_results = paste_bbox_img(img_results, main_camera_bbox_right_large, main_icon_large_right_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_camera_bbox_right_large, main_camera_right_icon_large, alpha=0.9)

        img_results = cv2.resize(img_results, (960, 540))
        cv2.imshow('display_main_right_large', img_results)


    # cv2.waitKey(0)

# ------------ display camera -------------
def display_camera():
    # img_path = "E:/gui_test/030.jpg"
    img_path = "E:/gui_test/001_000013.jpg"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (resize_W, resize_H))
    # iou = [0, 240, 0, 270]
    # iou = [0, test_w, 0, test_h]
    print(img.shape)
    # print(main_icon_small_bg.shape)
    # print(main_uterus_icon_small.shape)

    display_camera_left_unclick = True
    display_camera_right_unclick = False
    display_camera_left_click = False
    display_camera_right_click = True

    # ------------ display camera left un-click -------------
    if display_camera_left_unclick:
        img_results = paste_bbox_img(img, main_camera_bbox_left_large, camera_panel_large_left_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_camera_bbox_left_large, camera_panel_click_icon_large_left, alpha=0.9)
        img_results = paste_bbox_img(img_results, camera_panel_zoomin_bbox_left, camera_panel_small_left_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, camera_panel_zoomin_bbox_left, camera_panel_zoomin_icon_left, alpha=0.9)
        img_results = paste_bbox_img(img_results, camera_panel_zoomout_bbox_left, camera_panel_small_left_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, camera_panel_zoomout_bbox_left, camera_panel_zoomout_icon_left, alpha=0.9)
        img_results = paste_bbox_img(img_results, camera_panel_zoomauto_bbox_left, camera_panel_small_left_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, camera_panel_zoomauto_bbox_left, camera_panel_zoomauto_icon_left, alpha=0.9)
        img_results = paste_bbox_img(img_results, camera_panel_setzero_bbox_left, camera_panel_small_zero_left_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, camera_panel_setzero_bbox_left, camera_panel_setzero_icon_left, alpha=0.9)

        img_results = cv2.resize(img_results, (960, 540))
        cv2.imshow('display_camera_left_unclick', img_results)

    # ------------ display camera right un-click -------------
    if display_camera_right_unclick:
        img_results = paste_bbox_img(img, main_camera_bbox_right_large, camera_panel_large_right_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_camera_bbox_right_large, camera_panel_click_icon_large_right, alpha=0.9)
        img_results = paste_bbox_img(img_results, camera_panel_zoomin_bbox_right, camera_panel_small_right_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, camera_panel_zoomin_bbox_right, camera_panel_zoomin_icon_right, alpha=0.9)
        img_results = paste_bbox_img(img_results, camera_panel_zoomout_bbox_right, camera_panel_small_right_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, camera_panel_zoomout_bbox_right, camera_panel_zoomout_icon_right, alpha=0.9)
        img_results = paste_bbox_img(img_results, camera_panel_zoomauto_bbox_right, camera_panel_small_right_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, camera_panel_zoomauto_bbox_right, camera_panel_zoomauto_icon_right, alpha=0.9)
        img_results = paste_bbox_img(img_results, camera_panel_setzero_bbox_right, camera_panel_small_zero_right_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, camera_panel_setzero_bbox_right, camera_panel_setzero_icon_right, alpha=0.9)

        img_results = cv2.resize(img_results, (960, 540))
        cv2.imshow('display_camera_right_unclick', img_results)

    # ------------ display camera left click -------------
    if display_camera_left_click:
        img_results = paste_bbox_img(img, main_camera_bbox_left_large, camera_panel_large_left_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_camera_bbox_left_large, camera_panel_click_icon_large_left, alpha=0.9)
        img_results = paste_bbox_img(img_results, camera_panel_zoomin_bbox_left, camera_panel_small_left_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, camera_panel_zoomin_bbox_left, camera_panel_zoomin_click_icon_left, alpha=0.9)
        img_results = paste_bbox_img(img_results, camera_panel_zoomout_bbox_left, camera_panel_small_left_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, camera_panel_zoomout_bbox_left, camera_panel_zoomout_click_icon_left, alpha=0.9)
        img_results = paste_bbox_img(img_results, camera_panel_zoomauto_bbox_left, camera_panel_small_left_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, camera_panel_zoomauto_bbox_left, camera_panel_zoomauto_click_icon_left, alpha=0.9)
        img_results = paste_bbox_img(img_results, camera_panel_setzero_bbox_left, camera_panel_small_zero_left_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, camera_panel_setzero_bbox_left, camera_panel_setzero_click_icon_left, alpha=0.9)

        img_results = cv2.resize(img_results, (960, 540))
        cv2.imshow('display_camera_left_click', img_results)

    # ------------ display camera right click -------------
    if display_camera_right_click:
        img_results = paste_bbox_img(img, main_camera_bbox_right_large, camera_panel_large_right_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, main_camera_bbox_right_large, camera_panel_click_icon_large_right, alpha=0.9)
        img_results = paste_bbox_img(img_results, camera_panel_zoomin_bbox_right, camera_panel_small_right_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, camera_panel_zoomin_bbox_right, camera_panel_zoomin_click_icon_right, alpha=0.9)
        img_results = paste_bbox_img(img_results, camera_panel_zoomout_bbox_right, camera_panel_small_right_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, camera_panel_zoomout_bbox_right, camera_panel_zoomout_click_icon_right, alpha=0.9)
        img_results = paste_bbox_img(img_results, camera_panel_zoomauto_bbox_right, camera_panel_small_right_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, camera_panel_zoomauto_bbox_right, camera_panel_zoomauto_click_icon_right, alpha=0.9)
        img_results = paste_bbox_img(img_results, camera_panel_setzero_bbox_right, camera_panel_small_zero_right_bg, alpha=0.8)
        img_results = paste_bbox_img(img_results, camera_panel_setzero_bbox_right, camera_panel_setzero_click_icon_right, alpha=0.9)

        img_results = cv2.resize(img_results, (960, 540))
        cv2.imshow('display_camera_right_click', img_results)

    # img_results = cv2.resize(img_results, (960, 540))
    # cv2.imshow('img_camera', img_results)
    # cv2.waitKey(0)


# ------------ display uterus -------------
def display_uterus():
    # img_path = "E:/gui_test/030.jpg"
    img_path = "E:/gui_test/001_000013.jpg"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (resize_W, resize_H))
    # iou = [0, 240, 0, 270]
    # iou = [0, test_w, 0, test_h]
    print(img.shape)
    # print(main_icon_small_bg.shape)
    # print(main_uterus_icon_small.shape)

    display_uterus_unclick = True
    display_uterus_click = False
    uterus_left = False
    uterus_right = True

    # ------------ display uterus un-click -------------
    if display_uterus_unclick:

        # ------------- left --------------
        if uterus_left:
            img_results = paste_bbox_img(img, main_uterus_bbox_left_large, uterus_panel_large_left_bg, alpha=0.6)
            img_results = paste_bbox_img(img_results, main_uterus_bbox_left_large, uterus_panel_click_icon_large_left, alpha=0.8)

        # ------------- right --------------
        if uterus_right:
            img_results = paste_bbox_img(img, main_uterus_bbox_right_large, uterus_panel_large_right_bg, alpha=0.6)
            img_results = paste_bbox_img(img_results, main_uterus_bbox_right_large, uterus_panel_click_icon_large_right, alpha=0.8)

        img_results = paste_bbox_img(img_results, uterus_panel_up_bbox, uterus_panel_up_bg, alpha=0.6)
        img_results = paste_bbox_img(img_results, uterus_panel_up_bbox, uterus_panel_up_icon, alpha=0.8)
        img_results = paste_bbox_img(img_results, uterus_panel_down_bbox, uterus_panel_down_bg, alpha=0.6)
        img_results = paste_bbox_img(img_results, uterus_panel_down_bbox, uterus_panel_down_icon, alpha=0.8)
        img_results = paste_bbox_img(img_results, uterus_panel_left_bbox, uterus_panel_left_bg, alpha=0.6)
        img_results = paste_bbox_img(img_results, uterus_panel_left_bbox, uterus_panel_left_icon, alpha=0.8)
        img_results = paste_bbox_img(img_results, uterus_panel_right_bbox, uterus_panel_right_bg, alpha=0.6)
        img_results = paste_bbox_img(img_results, uterus_panel_right_bbox, uterus_panel_right_icon, alpha=0.8)
        img_results = paste_bbox_img(img_results, uterus_panel_insert_bbox, uterus_panel_insert_bg, alpha=0.6)
        img_results = paste_bbox_img(img_results, uterus_panel_insert_bbox, uterus_panel_insert_icon, alpha=0.8)
        img_results = paste_bbox_img(img_results, uterus_panel_retract_bbox, uterus_panel_retract_bg, alpha=0.6)
        img_results = paste_bbox_img(img_results, uterus_panel_retract_bbox, uterus_panel_retract_icon, alpha=0.8)

        img_results = cv2.resize(img_results, (960, 540))
        cv2.imshow('display_uterus_unclick', img_results)

    # ------------ display uterus click -------------
    if display_uterus_click:

        # ------------- left --------------
        if uterus_left:
            img_results = paste_bbox_img(img, main_uterus_bbox_left_large, uterus_panel_large_left_bg, alpha=0.6)
            img_results = paste_bbox_img(img_results, main_uterus_bbox_left_large, uterus_panel_click_icon_large_left, alpha=0.8)

        # ------------- right --------------
        if uterus_right:
            img_results = paste_bbox_img(img, main_uterus_bbox_right_large, uterus_panel_large_right_bg, alpha=0.6)
            img_results = paste_bbox_img(img_results, main_uterus_bbox_right_large, uterus_panel_click_icon_large_right, alpha=0.8)

        img_results = paste_bbox_img(img_results, uterus_panel_up_bbox, uterus_panel_up_bg, alpha=0.6)
        img_results = paste_bbox_img(img_results, uterus_panel_up_bbox, uterus_panel_up_click_icon, alpha=0.8)
        img_results = paste_bbox_img(img_results, uterus_panel_down_bbox, uterus_panel_down_bg, alpha=0.6)
        img_results = paste_bbox_img(img_results, uterus_panel_down_bbox, uterus_panel_down_click_icon, alpha=0.8)
        img_results = paste_bbox_img(img_results, uterus_panel_left_bbox, uterus_panel_left_bg, alpha=0.6)
        img_results = paste_bbox_img(img_results, uterus_panel_left_bbox, uterus_panel_left_click_icon, alpha=0.8)
        img_results = paste_bbox_img(img_results, uterus_panel_right_bbox, uterus_panel_right_bg, alpha=0.6)
        img_results = paste_bbox_img(img_results, uterus_panel_right_bbox, uterus_panel_right_click_icon, alpha=0.8)
        img_results = paste_bbox_img(img_results, uterus_panel_insert_bbox, uterus_panel_insert_bg, alpha=0.6)
        img_results = paste_bbox_img(img_results, uterus_panel_insert_bbox, uterus_panel_insert_click_icon, alpha=0.8)
        img_results = paste_bbox_img(img_results, uterus_panel_retract_bbox, uterus_panel_retract_bg, alpha=0.6)
        img_results = paste_bbox_img(img_results, uterus_panel_retract_bbox, uterus_panel_retract_click_icon, alpha=0.8)

        img_results = cv2.resize(img_results, (960, 540))
        cv2.imshow('display_uterus_click', img_results)

    # img_results = cv2.resize(img_results, (960, 540))
    # cv2.imshow('img_uterus', img_results)
    cv2.waitKey(0)


if __name__ == '__main__':
    display_main()
    display_camera()
    display_uterus()


