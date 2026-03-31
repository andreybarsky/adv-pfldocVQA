from attacks.masks import mask_include_all, mask_exclude_white, mask_bottom_right_corner, mask_sobel, mask_sobel_blue, mask_sobel_red, mask_sobel_chromatic, mask_edge_channelwise, mask_robust_sobel

DATAPATH = '/home/abarsky/data/advdoc_data_nsampl1000_nqst5.pkl'
MODEL_NAMES = ['pix2struct','donut']

AVAILABLE_MASKS = {
    "include_all": mask_include_all,
    "exclude_white": mask_exclude_white,
    "bottom_right_corner": mask_bottom_right_corner,
    "sobel": mask_sobel,
    "sobel_blue": mask_sobel_blue,
    "sobel_red": mask_sobel_red,
    "sobel_chromatic": mask_sobel_chromatic,
    "edge_channelwise": mask_edge_channelwise,
    "robust_sobel": mask_robust_sobel,
}
