# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
import torch
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator


class CityscapesEvaluator(DatasetEvaluator):
    """
    Base class for evaluation using cityscapes API.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): the name of the dataset.
                It must have the following metadata associated with it:
                "thing_classes", "gt_dir".
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._working_dir = tempfile.TemporaryDirectory(prefix="cityscapes_eval_")
        self._temp_dir = self._working_dir.name
        # All workers will write to the same results directory
        # TODO this does not work in distributed training
        self._temp_dir = comm.all_gather(self._temp_dir)[0]
        # print(f'j) in CityscapesEvaluator.reset,  comm.all_gather(self._temp_dir): {comm.all_gather(self._temp_dir)}')
        if self._temp_dir != self._working_dir.name:
            self._working_dir.cleanup()
        self._logger.info(
            "Writing cityscapes results to temporary directory {} ...".format(self._temp_dir)
        )


class CityscapesInstanceEvaluator(CityscapesEvaluator):
    """
    Evaluate instance segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """

    def process(self, inputs, outputs):

        # print('j) CityscapesInstanceEvaluator.process starts!!! ~~~~~~~~~~~')

        from cityscapesscripts.helpers.labels import name2label

        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_txt = os.path.join(self._temp_dir, basename + "_pred.txt")

            if "instances" in output:
                output = output["instances"].to(self._cpu_device)
                num_instances = len(output)
                with open(pred_txt, "w") as fout:
                    for i in range(num_instances):
                        pred_class = output.pred_classes[i]
                        classes = self._metadata.thing_classes[pred_class]
                        class_id = name2label[classes].id
                        score = output.scores[i]
                        mask = output.pred_masks[i].numpy().astype("uint8")
                        png_filename = os.path.join(
                            self._temp_dir, basename + "_{}_{}.png".format(i, classes)
                        )

                        Image.fromarray(mask * 255).save(png_filename)
                        fout.write(
                            "{} {} {}\n".format(os.path.basename(png_filename), class_id, score)
                        )
            else:
                # Cityscapes requires a prediction file for every ground truth image.
                with open(pred_txt, "w") as fout:
                    pass

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP" and "AP50".
        """
        
        # print('j) CityscapesInstanceEvaluator.evaluate starts!!! ~~~~~~~~~~~')

        comm.synchronize()
        if comm.get_rank() > 0:
            return
        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.gtInstancesFile = os.path.join(self._temp_dir, "gtInstances.json")

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_instanceIds.png"))
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(cityscapes_eval.getPrediction(gt, cityscapes_eval.args))
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )["averages"]

        ret = OrderedDict()
        ret["segm"] = {"AP": results["allAp"] * 100, "AP50": results["allAp50%"] * 100}
        self._working_dir.cleanup()
        return ret


class CityscapesSemSegEvaluator(CityscapesEvaluator):
    """
    Evaluate semantic segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """

    def process(self, inputs, outputs):
        
        # print('j) CityscapesSemSegEvaluator.process starts!!! ~~~~~~~~~~~')

        from cityscapesscripts.helpers.labels import trainId2label

        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_filename = os.path.join(self._temp_dir, basename + "_pred.png")

            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device).numpy() # i. TODO 이부분 조사아직못함. /21.3.27.21:30.
            pred = 255 * np.ones(output.shape, dtype=np.uint8) # i. 1 들로 채우네. 상관없나? 어차피 모든영역 다 그려주나? TODO output["sem_seg"] 가 뭔지 조사필요. /21.3.27.21:30.
            for train_id, label in trainId2label.items():
                if label.ignoreInEval:
                    continue
                pred[output == train_id] = label.id
            Image.fromarray(pred).save(pred_filename)

    def evaluate(self):
        
        # print('j) CityscapesSemSegEvaluator.evaluate starts!!! ~~~~~~~~~~~')

        comm.synchronize()
        if comm.get_rank() > 0:
            return
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)

        # i. 원소 하나 예시: ~~/cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelIds.png /21.3.28.9:48. 
        # groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_labelIds.png")) 
        # i. TODO 내플젝 할때는 위 한줄 코멘트아웃하고 아래의 내 코드 한줄을 살려줘야함. /21.3.26.21:51. 
        # i.21.3.24.20:34) 위 한줄을 바로아래한줄로 내가 변경해줬음. 잘 되나 모름. 해보는중.
        #  (cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py 에 설명 나오니까 보삼. 아직 제대로 안읽어봄.)
        #  암튼, 원래코드는 ~~labelIds.png 를 이용했는데, 이게 인스턴스id 가 아니고 카테고리id임.
        #  ~~labelTrainIds.png 도 마찬가지로 카테고리id인데, train 용도의 카테고리id 인것 뿐이니까 이거 사용해도 될것으로 생각됨.
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*_labelTrainIds.png"))

        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            # i.21.3.26.21:52) 왜 이걸써놨지..? 코드작성자의 실수인듯. 이렇게하면 바로위의 groundTruthImgList 랑 값 다를수있는데. 일단 걍 냅두자. 
            cityscapes_eval.args.groundTruthSearch
        )

        # i.21.3.28.12:12) 바로아래서 이용하는 cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling 의 getPrediction 함수 대신 사용해줄,
        #  내가만든 getPredictionJ 함수. (똑같은 일을 하는데, 내플젝에 맞게 만든거임. 하는일은 매우 단순.)
        def getPredictionJ(predPngDirpathJ, gtPngPathJ):
            gtPngFnameJ = os.path.basename(gtPngPathJ) # i. "~~_labelTrainIds.png"
            imgIdJ = gtPngFnameJ[:-len("_labelTrainIds.png")] # i. ex: "impAAA_BBB"
            predPngPathJ = glob.glob(os.path.join(predPngDirpathJ, imgIdJ+"*"))[0]
            print(f'j) predPngPathJ 예상: /임시/폴더의/경로/impAAA_BBB_pred.png') 
            print(f'j) predPngPathJ: {predPngPathJ}')
            return predPngPathJ

        predictionImgList = []
        for gt in groundTruthImgList:
            # print('j) gt in groundTruthImgList . . . . . . . . . . . . . . .')
            # i.21.3.24.21:14) ######################### 지금하고있는부분!! 코랩 돌리면 여기서 에러남!!!
            # i.21.3.28.11:49) getPrediction 함수가 하는일은 매우 간단함.
            #  gt png파일 경로를 넣어주면 그에 해당하는 프레딕션결과파일(죠위에 process에서 만들어준)의 경로를 리턴해주는것 뿐. 
            #  (죠 위의 process에서 프레딕션결과png파일들 만들어서 임시폴더에 저장해논상태고, 그 임시폴더의 경로가 cityscapes_eval.args.predictionPath 에 할당되어있지.) 
            #  getPrediction 함수의 두번째인풋 및 리턴값 예시: 
            #  gt ex: ~~/cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelIds.png 
            #  리턴값 ex: '/임시/폴더의/경로/frankfurt_000000_000294_leftImg8bit_pred.png' 
            #
            #  따라서, 내플젝 적용할땐, getPrediction 함수 쓸필요없고(내플젝의 파일명 구조는 cityscapes 플젝의 파일명 구조처럼 복잡하지 않으니), 
            #  걍 간단한 방법으로 gt(하나의 gt png파일의 경로)에 대한 프레딕션결과png파일의 경로를 찾아서 predictionImgList 에 append 해주면 됨.
            #  음.. 간단한로직이긴한데 걍 getPredictionJ 라는 함수 만들어서 해야겠다.
            #
            # predictionImgList.append(cityscapes_eval.getPrediction(cityscapes_eval.args, gt)) # TODO 내플젝돌릴땐 이거말고 아래의 코드(getPredictionJ 함수 이용하는) 사용해야함. /21.3.28.12:11.
            predictionImgList.append(getPredictionJ(cityscapes_eval.args.predictionPath, gt))
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )
        ret = OrderedDict()
        ret["sem_seg"] = {
            "IoU": 100.0 * results["averageScoreClasses"],
            "iIoU": 100.0 * results["averageScoreInstClasses"],
            "IoU_sup": 100.0 * results["averageScoreCategories"],
            "iIoU_sup": 100.0 * results["averageScoreInstCategories"],
        }
        self._working_dir.cleanup()
        return ret
