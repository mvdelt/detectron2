# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES
from detectron2.utils.file_io import PathManager

"""
This file contains functions to register the Cityscapes panoptic dataset to the DatasetCatalog.
"""


logger = logging.getLogger(__name__)


def get_cityscapes_panoptic_files(image_dir, gt_dir, json_info):
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    image_dict = {}
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        for basename in PathManager.ls(city_img_dir): # i. basename ex: "frankfurt_000000_000294_leftImg8bit.png"
            # i. image_file ex: "~~\leftImg8bit\val\frankfurt\frankfurt_000000_000294_leftImg8bit.png" (원 인풋이미지의 경로)
            image_file = os.path.join(city_img_dir, basename) 

            suffix = "_leftImg8bit.png"
            assert basename.endswith(suffix), basename
            basename = os.path.basename(basename)[: -len(suffix)] # i. basename ex: "frankfurt_000000_000294"

            image_dict[basename] = image_file # i. image_dict: map 'basename' to 'image_file'(path).

    for ann in json_info["annotations"]:
        # i. ann["image_id"] ex: "frankfurt_000000_000294" 
        #  (cityscapes데이터셋에서 각 annotation의 'image_id' 는 애시당초 위의 'basename' 과 동일하도록 만들어져있음. "frankfurt_000000_000294" 이런식.)
        image_file = image_dict.get(ann["image_id"], None) 
        assert image_file is not None, "No image {} found for annotation {}".format(
            ann["image_id"], ann["file_name"]
        )
        # i. ann["file_name"] ex: "frankfurt_000000_000294_gtFine_panoptic.png" (어노png 파일명)
        label_file = os.path.join(gt_dir, ann["file_name"]) # i. label_file ex: "~~\gtFine\cityscapes_panoptic_val\frankfurt_000000_000294_gtFine_panoptic.png"
        segments_info = ann["segments_info"]

        files.append((image_file, label_file, segments_info))

    assert len(files), "No images found in {}".format(image_dir)
    assert PathManager.isfile(files[0][0]), files[0][0]
    assert PathManager.isfile(files[0][1]), files[0][1]
    return files


def load_cityscapes_panoptic(image_dir, gt_dir, gt_json, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
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

    def _convert_category_id(segment_info, meta):
        # i.21.3.18.1:35) 지금 이 함수는 cityscapesscripts 의 createPanopticImgs.py 의 convert2panoptic 함수에서 
        #  useTrainId=False 로 적용했을때(segment_info 의 "category_id" 가 카테고리의 trainId 가 아닌 그냥id로 셋팅됨)를 가정하고 작동하는거네. 
        #  그래서결국, segment_info["category_id"] 를 카테고리의 그냥id에서 trainId 로 바꿔주는거임.
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
    ), "Please run `python cityscapesscripts/preparation/createPanopticImgs.py` to generate label files."  # noqa
    with open(gt_json) as f:
        json_info = json.load(f)


    # i. files: [(imgFilePath, labelPngFilePath, segments_info), (~~), ...] /21.3.10.18:34.
    #  files 의 각 튜플은 하나의 이미지에 대응됨.
    files = get_cityscapes_panoptic_files(image_dir, gt_dir, json_info) 
    ret = []
    for image_file, label_file, segments_info in files: # i. 각각 (하나의 이미지에 대응되는) 원본이미지파일경로, 어노png경로, segments_info(리스트). /21.3.10.18:34.
        # i. image_file ex: "~~\leftImg8bit\val\frankfurt\frankfurt_000000_000294_leftImg8bit.png"
        # i. label_file ex: "~~\gtFine\cityscapes_panoptic_val\frankfurt_000000_000294_gtFine_panoptic.png"
       
        # i. sem_label_file ex: "~~\gtFine\val\frankfurt\frankfurt_000000_000294_gtFine_labelTrainIds.png" 
        #  ->leftImg8bit 를 gtFine으로 바꿨는데, 파일명 뿐 아니라 폴더명에도 leftImg8bit 있어서 그것도 gtFine 으로 바꼈음!!! 이거모르고 틀린줄알았네;;;/21.3.12.8:00.
        sem_label_file = (
            image_file.replace("leftImg8bit", "gtFine").split(".")[0] + "_labelTrainIds.png"  
        )
        segments_info = [_convert_category_id(x, meta) for x in segments_info]

        ret.append(
            # i.21.3.12.8:19) 결국 요게 핵심. 
            #  지금 이 함수(load_cityscapes_panoptic)의 목적이 이 딕셔너리들의 리스트를 반환하는거니까.
            {
                "file_name": image_file,
                # i. "image_id" ex: 'frankfurt_000000_000294' /21.3.12.15:23.
                "image_id": "_".join(
                    os.path.splitext(os.path.basename(image_file))[0].split("_")[:3]
                ),
                # i. 아. 요 "sem_seg_file_name" 이 필요해서 Det2 문서 'Use Builtin Datasets'에서 cityscapes 데이터셋 준비해주는부분에서
                #  createTrainIdLabelImgs.py 를 실행해서 ~~labelTrainIds.png 들을 만들어줬던거구나. 
                #  난 ~~instanceIds.png 들만 잇으면 되는데 굳이 ~~labelTrainIds.png 는 왜 만들어줬나 했네. Det2 문서가 잘못된건줄 알앗네ㅋ.
                #  ~~labelTrainIds.png 가 (id값들을 트레이닝용 id값들로 변환해준) semantic seg 어노png에 해당하는거니까. 
                #  근데..어쨋든 "pan_seg_file_name" 어노png에 모든 정보는 다 들어있는데, 굳이 "sem_seg_file_name" 은 왜 필요..??/21.3.10.18:49
                # i.21.3.10.23:38) TODO Q: ->뭐지?? 지금 이 함수(load_cityscapes_panoptic)
                #  에 대응되는 COCO 함수인 load_coco_panoptic_json 에서는 "sem_seg_file_name" 정보 안넣어주는데???
                "sem_seg_file_name": sem_label_file, 
                # i. 지금 label_file 은 COCO panoptic 형식의 어노png 파일인데(cityscapesscripts 코드에서 COCO panoptic 형식으로 변환해줌), 
                #  Det2 형식의 "pan_seg_file_name"도 어차피 COCO panoptic 과 동일한 형식의 어노png(id값을 256진법으로 RGB로 변환한거. 
                #  Det2 문서의 "pan_seg_file_name" 설명에 나오는 panopticapi.utils.id2rgb 함수가 해주는게 그거임.)를 받기때문에,
                #  아래처럼 그냥 그대로 할당해줘도 됨./21.3.10.19:56.
                "pan_seg_file_name": label_file, 
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


_RAW_CITYSCAPES_PANOPTIC_SPLITS = {
    "cityscapes_fine_panoptic_train": (
        "cityscapes/leftImg8bit/train", # i. 원 인풋이미지들 있는 디렉토리./21.3.10.12:02.
        "cityscapes/gtFine/cityscapes_panoptic_train", # i. COCO형식으로 변환된 어노png파일들 있는 디렉토리./21.3.10.12:02.
        "cityscapes/gtFine/cityscapes_panoptic_train.json", # i. COCO형식으로 변환된 어노json파일 경로./21.3.10.12:02.
    ),
    "cityscapes_fine_panoptic_val": (
        "cityscapes/leftImg8bit/val", # i. 원 인풋이미지들 있는 디렉토리./21.3.10.12:02.
        "cityscapes/gtFine/cityscapes_panoptic_val", # i. COCO형식으로 변환된 어노png파일들 있는 디렉토리./21.3.10.12:02.
        "cityscapes/gtFine/cityscapes_panoptic_val.json", # i. COCO형식으로 변환된 어노json파일 경로./21.3.10.12:02.
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
    thing_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    thing_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]
    stuff_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    stuff_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]

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

    for k in CITYSCAPES_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (image_dir, gt_dir, gt_json) in _RAW_CITYSCAPES_PANOPTIC_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)

        DatasetCatalog.register(
            key, lambda x=image_dir, y=gt_dir, z=gt_json: load_cityscapes_panoptic(x, y, z, meta)
        )
        MetadataCatalog.get(key).set(
            panoptic_root=gt_dir, # i. COCO형식으로 변환된 어노png파일들 있는 디렉토리./21.3.10.12:02.
            image_root=image_dir, # i. 원 인풋이미지들 있는 디렉토리./21.3.10.12:02.
            panoptic_json=gt_json, # i. COCO형식으로 변환된 어노json파일 경로./21.3.10.12:02.
            # i. COCO형식으로 변환하기 전의, cityscapes형식의 어노png, 어노json (~~instanceIds.png, ~~polygons.json 등) 들이 있는 디렉토리./21.3.10.12:08.
            gt_dir=gt_dir.replace("cityscapes_panoptic_", ""), 
            evaluator_type="cityscapes_panoptic_seg",
            ignore_label=255,
            label_divisor=1000,
            **meta,
        )
