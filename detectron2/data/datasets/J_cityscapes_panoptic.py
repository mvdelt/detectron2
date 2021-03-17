

##############################################################################################################
#
# i.21.3.10.23:44) Det2 의 cityscapes_panoptic.py 파일의 
#  register_all_cityscapes_panoptic, load_cityscapes_panoptic 등의 함수들을 
#  내panoptic플젝에 맞게 수정해주려고 만든 파일.# 
#  TODO Q: 근데 Det2 설치할때 이렇게 내가 새로 추가해준 파일은 어케되는거지..?
#
# i.21.3.11.10:27) 알고있듯이 builtin.py 에서 이것저것 기본적인것들 다 레지스터해주는데
#  이때 register_all_cityscapes_panoptic 도 레지스터해주는데,
#  지금 내 플젝에선 TODO 내가 따로 호출해서 레지스터 해주자.
#
##############################################################################################################



# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES    # i.21.3.11.00:00) ->이거쓰면 안됨. 내플젝이라.
from detectron2.utils.file_io import PathManager

from collections import namedtuple

"""
This file contains functions to register the Cityscapes panoptic dataset to the DatasetCatalog.
"""


# # All Cityscapes categories, together with their nice-looking visualization colors
# # It's from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py  # noqa
# CITYSCAPES_CATEGORIES = [
#     {"color": (128, 64, 128), "isthing": 0, "id": 7, "trainId": 0, "name": "road"},
#     {"color": (244, 35, 232), "isthing": 0, "id": 8, "trainId": 1, "name": "sidewalk"},
#     {"color": (70, 70, 70), "isthing": 0, "id": 11, "trainId": 2, "name": "building"},
#     {"color": (102, 102, 156), "isthing": 0, "id": 12, "trainId": 3, "name": "wall"},
#     {"color": (190, 153, 153), "isthing": 0, "id": 13, "trainId": 4, "name": "fence"},
#     {"color": (153, 153, 153), "isthing": 0, "id": 17, "trainId": 5, "name": "pole"},
#     {"color": (250, 170, 30), "isthing": 0, "id": 19, "trainId": 6, "name": "traffic light"},
#     {"color": (220, 220, 0), "isthing": 0, "id": 20, "trainId": 7, "name": "traffic sign"},
#     {"color": (107, 142, 35), "isthing": 0, "id": 21, "trainId": 8, "name": "vegetation"},
#     {"color": (152, 251, 152), "isthing": 0, "id": 22, "trainId": 9, "name": "terrain"},
#     {"color": (70, 130, 180), "isthing": 0, "id": 23, "trainId": 10, "name": "sky"},
#     {"color": (220, 20, 60), "isthing": 1, "id": 24, "trainId": 11, "name": "person"},
#     {"color": (255, 0, 0), "isthing": 1, "id": 25, "trainId": 12, "name": "rider"},
#     {"color": (0, 0, 142), "isthing": 1, "id": 26, "trainId": 13, "name": "car"},
#     {"color": (0, 0, 70), "isthing": 1, "id": 27, "trainId": 14, "name": "truck"},
#     {"color": (0, 60, 100), "isthing": 1, "id": 28, "trainId": 15, "name": "bus"},
#     {"color": (0, 80, 100), "isthing": 1, "id": 31, "trainId": 16, "name": "train"},
#     {"color": (0, 0, 230), "isthing": 1, "id": 32, "trainId": 17, "name": "motorcycle"},
#     {"color": (119, 11, 32), "isthing": 1, "id": 33, "trainId": 18, "name": "bicycle"},
# ]

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.   # <-inverse mapping에선 아래 labels리스트의 (동일한 trainId를 가지는 Label들 중)1번째녀석을 사용한다고 되어있지./i.21.3.5.18:58.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    # i.21.3.5.17:19) ignoreInEval 값이 True 면 evaluation(cityscapes 대회서버에서 하는 이밸류에이션)에 반영 안됨!!!
    #  그래서 train시에만 자유롭게 정해서 이용하라고 trainId 가 있는거고, 
    #  요아래 Label 들의 trainId 값들은 대회측에서 일단 기본값으로 정해논건데(맘대로바꿀수있음), 
    #  ignoreInEval 이 True 인 놈들에 대해서는 trainId 값을 255나 -1등으로 해놓은듯.
    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


# # i.21.3.11.00:00) cityscapesscripts 의 labels.py 에 내가 내플젝에맞게 새로 정해준 'labels' 복붙함.
# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled_Label'      ,  0 ,        0 , 'voidJ'           , 0       , False        , False        , (  0,  0,  0) ),
#     Label(  'Rt_sinus'             ,  1 ,        1 , 'sinusJ'          , 1       , False        , False        , (  0,  0,255) ),
#     Label(  'Lt_sinus'             ,  2 ,        2 , 'sinusJ'          , 1       , False        , False        , (255,  0,  0) ),
#     Label(  'maxilla'              ,  3 ,        3 , 'boneJ'           , 2       , False        , False        , (162,156,255) ),
#     Label(  'mandible'             ,  4 ,        4 , 'boneJ'           , 2       , False        , False        , (185,181,247) ),
#     Label(  'Rt_canal'             ,  5 ,        5 , 'canalJ'          , 3       , False        , False        , ( 76, 68,212) ),
#     Label(  'Lt_canal'             ,  6 ,        6 , 'canalJ'          , 3       , False        , False        , (194, 37,144) ),
#     Label(  't_normal'             ,  7 ,        7 , 'toothJ'          , 4       , True         , False        , ( 66,158, 27) ),
#     Label(  't_tx'                 ,  8 ,        8 , 'toothJ'          , 4       , True         , False        , ( 88,214, 34) ),
#     Label(  'impl'                 ,  9 ,        9 , 'toothJ'          , 4       , True         , False        , (116,255, 56) ),
# ]

# i.21.3.17.1:04) Rt, Lt 구분 없앤걸로 수정.(걍 귀찮아서 Det2 내장 좌우플립 데이터오그멘테이션 사용해주기 위해서 sinus 랑 canal 의 좌우구분 없애버림.)
# i.21.3.17.17:38) TODO: sinus, canal 의 hasInstances 를 True 로 해줘야하나???
# i.21.3.17.19:00) TODO: unlabeled_Label 의 trainId 를 255로 해주는게 나은가??? 그리고나서 모든걸 trainId 기준으로 준비해주고..??
#  (난 지금은 id랑 trainId 가 동일해서 그냥 크게 구분없이 해줫을거임. )
#    아니, 걍 unlabeled_Label 자체를 걍 없애버리고 총 클래스수를 8개가 아닌 7개로 해주면 될것같은데?? 
#  panoptic deeplab 에서도 coco 는 133개, cityscapes 는 19개로 해줫는데, 모두 unlabeled 는 빼고 실제로 의미잇는 클래스들 갯수만 센거임.
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled_Label'      ,  0 ,        0 , 'voidJ'           , 0       , False        , False        , (  0,  0,  0) ),
    Label(  'sinus'                ,  1 ,        1 , 'sinusJ'          , 1       , False        , False        , (  0,  0,255) ),
    Label(  'maxilla'              ,  2 ,        2 , 'boneJ'           , 2       , False        , False        , (162,156,255) ),
    Label(  'mandible'             ,  3 ,        3 , 'boneJ'           , 2       , False        , False        , (185,181,247) ),
    Label(  'canal'                ,  4 ,        4 , 'canalJ'          , 3       , False        , False        , ( 76, 68,212) ),
    Label(  't_normal'             ,  5 ,        5 , 'toothJ'          , 4       , True         , False        , ( 66,158, 27) ),
    Label(  't_tx'                 ,  6 ,        6 , 'toothJ'          , 4       , True         , False        , ( 88,214, 34) ),
    Label(  'impl'                 ,  7 ,        7 , 'toothJ'          , 4       , True         , False        , (116,255, 56) ),
]

# i.21.3.10.23:59) 기존 CITYSCAPES_CATEGORIES 의 형태로 변경해줌.
#  일단 CITYSCAPES_CATEGORIES 라는 변수명 유지하면서 _J 만 붙여줬음.
CITYSCAPES_CATEGORIES_J = \
[{"color": label.color, "isthing": 1 if label.hasInstances else 0, "id": label.id, "trainId": label.trainId, "name": label.name} for label in labels]




logger = logging.getLogger(__name__)


# i.21.3.11.00:15) 
#  TODO: 1) 내플젝은 city가 없음. 
#        2) cityscapes데이터셋의 어노json의 각 annotation의 'image_id' 는 애시당초 'basename' 과 동일하도록 만들어져있음. 
#           "frankfurt_000000_000294" 이런식. 나는 이걸 지금 A00B 뭐 이런식으로 해논상태인거지. imp2_45.jpg 뭐이런걸 2045 이런식으로. 
#           굳이그렇게바꿔줄필요없이 그냥 "imp2_45" 이렇게 해줘도 됨.
#           그리고 "_leftImg8bit.png" 관련된부분들도 수정하고. 예를들어 난 "_leftImg8bit.png" 을 제거해준다거나 할 필요가 없지.

# def get_cityscapes_panoptic_files(image_dir, gt_dir, json_info):
def get_cocoFormAnnosPerImg_list(image_dir, gt_dir, json_info): # i.21.3.12.18:58) 함수명 get_cityscapes_panoptic_files 에서 내플젝에맞게 이름변경.
    """
    # i.21.3.11.23:28) 이 함수의 목적은,
    #  한 [원본인풋이미지]에 해당하는 [어노png] 와 [어노json에기록돼있는 segments_info] 이렇게 3가지를 뽑아내서 내뱉어주는것.
    #  image_dir 에 있는 원본 인풋이미지, gt_dir 에 있는 어노png, json_info 의 (각 이미지에 해당하는)annotation의 segments_info
    #  이렇게 3가지를 튜플로 만들어서(한 이미지당 한 튜플인거지) 리스트로 만들어서 반환함.
    """

    # i.21.3.12.18:30)
    # image_dir ex:             "someRootJ/panopticSeg_dentPanoJ/inputOriPano/train", # i. 원 인풋이미지들 있는 디렉토리./21.3.10.12:02.
    # gt_dir ex:                "someRootJ/panopticSeg_dentPanoJ/gt/J_cocoformat_panoptic_train", # i. COCO형식으로 변환된 어노png파일들 있는 디렉토리./21.3.10.12:02.
    # json_info ex: loaded from "someRootJ/panopticSeg_dentPanoJ/gt/J_cocoformat_panoptic_train.json", # i. COCO형식으로 변환된 어노json파일 경로./21.3.10.12:02.


    files = []
    # scan through the directory
    # i. image_dir ex: "someRootJ\panopticSeg_dentPanoJ\inputOriPano\train" /21.3.12.16:06.
    

    # i. 21.3.12.16:10) 내플젝에선 city들 없어서 일케해줄필요없음. 
    # cities = PathManager.ls(image_dir)
    # logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    # image_dict = {}
    # for city in cities:
    #     city_img_dir = os.path.join(image_dir, city)
    #     for basename in PathManager.ls(city_img_dir): # i. basename ex: "frankfurt_000000_000294_leftImg8bit.png"
    #         # i. image_file ex: "~~\leftImg8bit\val\frankfurt\frankfurt_000000_000294_leftImg8bit.png" (원 인풋이미지의 경로)
    #         image_file = os.path.join(city_img_dir, basename) 

    #         suffix = "_leftImg8bit.png"
    #         assert basename.endswith(suffix), basename
    #         basename = os.path.basename(basename)[: -len(suffix)] # i. basename ex: "frankfurt_000000_000294"

    #         image_dict[basename] = image_file # i. image_dict: map 'basename' to 'image_file'(path).  (여기서 'basename'이 이미지id 랑 같은거임)


    # i.21.3.12.16:17) 기존코드와 마찬가지로, 이미지id -> 이미지파일경로 맵핑딕셔너리 만듦.
    #  원본인풋이미지 파일명에서 이미지id(바로위 기존코드의 'basename'에 해당) 추출.
    imgId2imgFilePathJ = {}
    for imgFileJ in os.listdir(image_dir):
        # i. imgFileJ ex: "imp2_1.jpg"
        img_id = os.path.splitext(imgFileJ)[0]
        imgId2imgFilePathJ[img_id] = os.path.join(image_dir, imgFileJ)


    for ann in json_info["annotations"]:
        # i. ann["image_id"] ex: "frankfurt_000000_000294" 
        #  (cityscapes데이터셋에서 각 annotation의 'image_id' 는 애시당초 위의 'basename' 과 동일하도록 만들어져있음. "frankfurt_000000_000294" 이런식.)
        imgFilePath = imgId2imgFilePathJ.get(ann["image_id"], None) # i. <-원래 변수명 image_dict 였는데 내가 imgId2imgFilePathJ 로 바꿨음. 의미전달상.
        assert imgFilePath is not None, "No image {} found for annotation {}".format(
            ann["image_id"], ann["file_name"]
        )
        # i. ann["file_name"] ex: "frankfurt_000000_000294_gtFine_panoptic.png" (어노png 파일명. cityscapes 데이터셋의경우.)
        # i. ann["file_name"] ex: "imp2_1_panopticAnno.png" (어노png 파일명. 내플젝의경우.)
        cocoAnnoPngPath = os.path.join(gt_dir, ann["file_name"]) 
        # i. ->cocoAnnoPngPath ex:                            "~~\gtFine\cityscapes_panoptic_val\frankfurt_000000_000294_gtFine_panoptic.png" (cityscapes 데이터셋의경우.)
        # i. ->cocoAnnoPngPath ex: "someRootJ\panopticSeg_dentPanoJ\gt\J_cocoformat_panoptic_train\imp2_1_panopticAnno.png" (내플젝의경우.)
        cocoJson_segments_info = ann["segments_info"]

        files.append((imgFilePath, cocoAnnoPngPath, cocoJson_segments_info))

    assert len(files), "No images found in {}".format(image_dir)
    assert PathManager.isfile(files[0][0]), files[0][0]
    assert PathManager.isfile(files[0][1]), files[0][1]
    return files




# def load_cityscapes_panoptic(image_dir, gt_dir, gt_json, meta):
def load_panopticSeg_dentPanoJ(image_dir, gt_dir, gt_json, meta): # i.21.3.12.20:37) 내플젝에맞게 함수명 변경함.
    """
    Args:

        image_dir (str): path to the raw dataset. e.g., 
            "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train".
        gt_json (str): path to the json file. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train.json".

        meta (dict): dictionary containing "thing_dataset_id_to_contiguous_id"
            and "stuff_dataset_id_to_contiguous_id" to map category ids to
            contiguous ids for training.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    # i.21.3.12.18:30)
    # image_dir ex: "someRootJ/panopticSeg_dentPanoJ/inputOriPano/train", # i. 원 인풋이미지들 있는 디렉토리./21.3.10.12:02.
    # gt_dir ex:    "someRootJ/panopticSeg_dentPanoJ/gt/J_cocoformat_panoptic_train", # i. COCO형식으로 변환된 어노png파일들 있는 디렉토리./21.3.10.12:02.
    # gt_json ex:   "someRootJ/panopticSeg_dentPanoJ/gt/J_cocoformat_panoptic_train.json", # i. COCO형식으로 변환된 어노json파일 경로./21.3.10.12:02.


    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        return segment_info

    assert os.path.exists(
        gt_json
    ), "j) Please run `cityscapesscripts_mvdeltGithub\cityscapesscripts\preparation\J_createPanopticImgs.py` to generate label files."  # noqa

    with open(gt_json) as f:
        json_info = json.load(f)

    
    # i. files: [(imgFilePath, labelPngFilePath, segments_info), (~~), ...] /21.3.10.18:34.
    #  files 의 각 튜플은 하나의 이미지에 대응됨.
    files = get_cocoFormAnnosPerImg_list(image_dir, gt_dir, json_info) # i.21.3.12.18:58) <-함수명 get_cityscapes_panoptic_files 에서 내플젝에맞게 이름변경.
    ret = []
    for imgFilePath, cocoAnnoPngPath, segments_info in files: # i. 각각 (하나의 이미지에 대응되는) 원본이미지파일경로, 어노png경로, segments_info(리스트). /21.3.10.18:34.
        # i. imgFilePath ex: "someRootJ\panopticSeg_dentPanoJ\inputOriPano\train\imp2_1.jpg" /21.3.12.15:36.
        # i. cocoAnnoPngPath ex: "someRootJ\panopticSeg_dentPanoJ\gt\J_cocoformat_panoptic_train\imp2_1_panopticAnno.png" /21.3.12.15:37.
              
        # i. sem_label_file ex: "someRootJ\panopticSeg_dentPanoJ\gt\train\imp2_1_labelTrainIds.png"
        sem_label_file = (
            imgFilePath.replace("inputOriPano", "gt").split(".")[0] + "_labelTrainIds.png"  
        )
        segments_info = [_convert_category_id(x, meta) for x in segments_info]

        ret.append(
            # i.21.3.12.8:19) 결국 요게 핵심. 
            #  지금 이 함수(load_cityscapes_panoptic)의 목적이 이 딕셔너리들의 리스트를 반환하는거니까.
            {
                "file_name": imgFilePath,
                # i. "image_id" ex: 'imp2_1' /21.3.12.15:23.
                "image_id": os.path.splitext(os.path.basename(imgFilePath))[0],
                # i. 아. 요 "sem_seg_file_name" 이 필요해서 Det2 문서 'Use Builtin Datasets'에서 cityscapes 데이터셋 준비해주는부분에서
                #  createTrainIdLabelImgs.py 를 실행해서 ~~labelTrainIds.png 들을 만들어줬던거구나. 
                #  난 ~~instanceIds.png 들만 잇으면 되는데 굳이 ~~labelTrainIds.png 는 왜 만들어줬나 했네. Det2 문서가 잘못된건줄 알앗네ㅋ.
                #  ~~labelTrainIds.png 가 (id값들을 트레이닝용 id값들로 변환해준) semantic seg 어노png에 해당하는거니까. 
                #  근데..어쨋든 "pan_seg_file_name" 어노png에 모든 정보는 다 들어있는데, 굳이 "sem_seg_file_name" 은 왜 필요..??/21.3.10.18:49
                # i.21.3.10.23:38) TODO Q: ->뭐지?? 지금 이 함수(load_cityscapes_panoptic)
                #  에 대응되는 COCO 함수인 load_coco_panoptic_json 에서는 "sem_seg_file_name" 정보 안넣어주는데???
                "sem_seg_file_name": sem_label_file, 
                # i. 지금 cocoAnnoPngPath 은 COCO panoptic 형식의 어노png 파일인데(cityscapesscripts 코드에서 COCO panoptic 형식으로 변환해줌), 
                #  Det2 형식의 "pan_seg_file_name"도 어차피 COCO panoptic 과 동일한 형식의 어노png(id값을 256진법으로 RGB로 변환한거. 
                #  Det2 문서의 "pan_seg_file_name" 설명에 나오는 panopticapi.utils.id2rgb 함수가 해주는게 그거임.)를 받기때문에,
                #  아래처럼 그냥 그대로 할당해줘도 됨./21.3.10.19:56.
                "pan_seg_file_name": cocoAnnoPngPath, 
                "segments_info": segments_info, 
            }
        )


    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    assert PathManager.isfile(
        ret[0]["pan_seg_file_name"]
    ), "Please generate panoptic annotation with python cityscapesscripts/preparation/createPanopticImgs.py"  # noqa

    return ret


# _RAW_CITYSCAPES_PANOPTIC_SPLITS = {
#     "cityscapes_fine_panoptic_train": (
#         "cityscapes/leftImg8bit/train", # i. 원 인풋이미지들 있는 디렉토리./21.3.10.12:02.
#         "cityscapes/gtFine/cityscapes_panoptic_train", # i. COCO형식으로 변환된 어노png파일들 있는 디렉토리./21.3.10.12:02.
#         "cityscapes/gtFine/cityscapes_panoptic_train.json", # i. COCO형식으로 변환된 어노json파일 경로./21.3.10.12:02.
#     ),
#     "cityscapes_fine_panoptic_val": (
#         "cityscapes/leftImg8bit/val", # i. 원 인풋이미지들 있는 디렉토리./21.3.10.12:02.
#         "cityscapes/gtFine/cityscapes_panoptic_val", # i. COCO형식으로 변환된 어노png파일들 있는 디렉토리./21.3.10.12:02.
#         "cityscapes/gtFine/cityscapes_panoptic_val.json", # i. COCO형식으로 변환된 어노json파일 경로./21.3.10.12:02.
#     ),
#     # "cityscapes_fine_panoptic_test": not supported yet
# }


# i.21.3.11.12:57) TODO: 지금 여기 하다말앗음!! 내플젝에맞게 폴더구조 변경해주고있음. 폴더명 바꿔주고있고.
#  점점 cityscapes 데이터셋 폴더구조와 비슷해지네ㅋ.
#  띄어쓰기해놓은곳 우측부분들이 아직 수정 못해준것들임.
# i.21.3.13.0:32) train 관련해선 뭐 대충 일케하면 될듯한데, val 관련해서는 아직 안만들어준 상태임.
#
# i.21.3.12.20:41) 변수명 기존 _RAW_CITYSCAPES_PANOPTIC_SPLITS 에서 내플젝에맞게 변경함.
_RAW_PANOPTICSEG_DENTPANO_SPLITS_J = {
    "panopticSeg_dentPanoJ_train": (
        "panopticSeg_dentPanoJ/inputOriPano/train", # i. 원 인풋이미지들 있는 디렉토리./21.3.10.12:02.
        "panopticSeg_dentPanoJ/gt/J_cocoformat_panoptic_train", # i. COCO형식으로 변환된 어노png파일들 있는 디렉토리./21.3.10.12:02.
        "panopticSeg_dentPanoJ/gt/J_cocoformat_panoptic_train.json", # i. COCO형식으로 변환된 어노json파일 경로./21.3.10.12:02.
    ),
    # i.21.3.11.22:56) TODO: val 관련된것들은 아직 안만들어준 상태임.
    "panopticSeg_dentPanoJ_val": (
        "panopticSeg_dentPanoJ/inputOriPano/val", # i. 원 인풋이미지들 있는 디렉토리./21.3.10.12:02.
        "panopticSeg_dentPanoJ/gt/J_cocoformat_panoptic_val", # i. COCO형식으로 변환된 어노png파일들 있는 디렉토리./21.3.10.12:02.
        "panopticSeg_dentPanoJ/gt/J_cocoformat_panoptic_val.json", # i. COCO형식으로 변환된 어노json파일 경로./21.3.10.12:02.
    ),
    # "cityscapes_fine_panoptic_test": not supported yet
}


def register_all_cityscapes_panoptic(root):
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in CITYSCAPES_CATEGORIES_J]
    thing_colors = [k["color"] for k in CITYSCAPES_CATEGORIES_J]
    stuff_classes = [k["name"] for k in CITYSCAPES_CATEGORIES_J]
    stuff_colors = [k["color"] for k in CITYSCAPES_CATEGORIES_J]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # There are three types of ids in cityscapes panoptic segmentation:
    # (1) category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the classifier
    # (2) instance id: this id is used to differentiate different instances from
    #   the same category. For "stuff" classes, the instance id is always 0; for
    #   "thing" classes, the instance id starts from 1 and 0 is reserved for
    #   ignored instances (e.g. crowd annotation).            
    #
    # i.21.3.10.11:45) ->위 설명(2) 틀렸음. stuff들은 instance id가 아예 존재하지 않고, 
    #  thing들중 instance 구분될경우(ex: car들을 각각 구분해서 어노테이션한경우) instance id 는 0부터 시작임.
    #  thing들중 instance 구분안될경우(ex: car들이 그룹으로 뭉쳐있어서 뭉뚱그려서 어노테이션한경우) stuff와 마찬가지로 instance id가 아예 존재x.
    #  cityscapesscripts 의 json2instanceImg.py 의 첫부분 코멘트 참고.
    #
    # (3) panoptic id: this is the compact id that encode both category and
    #   instance id by: category_id * 1000 + instance_id.     
    # 
    # # i.21.3.10.11:51) ->위 설명(3) stuff들 및 instance 구분안되는 thing들에는 해당x. 얘넨 그냥 카테고리id만 달랑 있음.
    #  cityscapesscripts 의 json2instanceImg.py 의 첫부분 코멘트 참고.
    #
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in CITYSCAPES_CATEGORIES_J:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (image_dir, gt_dir, gt_json) in _RAW_PANOPTICSEG_DENTPANO_SPLITS_J.items(): # i.21.3.12.20:41) 변수명 기존 _RAW_CITYSCAPES_PANOPTIC_SPLITS 에서 내플젝에맞게 변경함.
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)

        DatasetCatalog.register(
            # key, lambda x=image_dir, y=gt_dir, z=gt_json: load_cityscapes_panoptic(x, y, z, meta)
            key, lambda x=image_dir, y=gt_dir, z=gt_json: load_panopticSeg_dentPanoJ(x, y, z, meta) # i.21.3.12.20:37) 내플젝에맞게 함수명 변경함.
        )
        MetadataCatalog.get(key).set(
            panoptic_root=gt_dir, # i. COCO형식으로 변환된 어노png파일들 있는 디렉토리./21.3.10.12:02.
            image_root=image_dir, # i. 원 인풋이미지들 있는 디렉토리./21.3.10.12:02.
            panoptic_json=gt_json, # i. COCO형식으로 변환된 어노json파일 경로./21.3.10.12:02.
            # i. COCO형식으로 변환하기 전의, cityscapes형식의 어노png, 어노json (~~instanceIds.png, ~~polygons.json 등) 들이 있는 디렉토리./21.3.10.12:08.
            gt_dir=gt_dir.replace("J_cocoformat_panoptic_", ""), # i. 내플젝에맞게 수정함. /21.3.12.20:56. 
            ################### 내플젝에맞게 수정필요.
            # i.21.3.12.23:51) ->이밸류에이터 관련 코드들 좀 살펴봣는데, 일단 걍 이대로("cityscapes_panoptic_seg"로) 냅둬도될듯.. 코드 좀 더 뜯어봐야함!!
            #  대신, Det2 인스톨시에 내가수정한거 적용되게 해서, 내플젝에맞게 변경한 labels.py 가 적용되어야할듯!!!
            #  그 외에 또 고려해줄거 잇으려나..?
            evaluator_type="cityscapes_panoptic_seg", 
            ################### 내플젝에맞게 수정필요.
            # i. ->현재 내플젝에선 무시하는 카테고리가 없는상태라 이대로(255로) 냅둬도 상관은 없을듯.
            ignore_label=255, 
            label_divisor=1000,
            **meta,
        )
