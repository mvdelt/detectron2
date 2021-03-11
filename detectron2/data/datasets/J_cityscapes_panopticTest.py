
from collections import namedtuple


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



# i.21.3.11.00:00) cityscapesscripts 의 labels.py 에 내가 내플젝에맞게 새로 정해준 'labels' 복붙함.
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled_Label'      ,  0 ,        0 , 'voidJ'           , 0       , False        , False        , (  0,  0,  0) ),
    Label(  'Rt_sinus'             ,  1 ,        1 , 'sinusJ'          , 1       , False        , False        , (  0,  0,255) ),
    Label(  'Lt_sinus'             ,  2 ,        2 , 'sinusJ'          , 1       , False        , False        , (255,  0,  0) ),
    Label(  'maxilla'              ,  3 ,        3 , 'boneJ'           , 2       , False        , False        , (162,156,255) ),
    Label(  'mandible'             ,  4 ,        4 , 'boneJ'           , 2       , False        , False        , (185,181,247) ),
    Label(  'Rt_canal'             ,  5 ,        5 , 'canalJ'          , 3       , False        , False        , ( 76, 68,212) ),
    Label(  'Lt_canal'             ,  6 ,        6 , 'canalJ'          , 3       , False        , False        , (194, 37,144) ),
    Label(  't_normal'             ,  7 ,        7 , 'toothJ'          , 4       , True         , False        , ( 66,158, 27) ),
    Label(  't_tx'                 ,  8 ,        8 , 'toothJ'          , 4       , True         , False        , ( 88,214, 34) ),
    Label(  'impl'                 ,  9 ,        9 , 'toothJ'          , 4       , True         , False        , (116,255, 56) ),
]

# i.21.3.10.23:59) 기존 CITYSCAPES_CATEGORIES 의 형태로 변경해줌.
#  일단 걍 CITYSCAPES_CATEGORIES 라는 변수명 그대로 이용해주자. 내플젝이지만.
CITYSCAPES_CATEGORIES = \
[{"color": label.color, "isthing": 1 if label.hasInstances else 0, "id": label.id, "trainId": label.trainId, "name": label.name} for label in labels]



print(f'j) type labels[0]: {type(labels[0])}')
print(f'j) CITYSCAPES_CATEGORIES: [')
for d in CITYSCAPES_CATEGORIES: print(d)
print(f']')