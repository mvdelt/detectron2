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

            #  COCOPanopticEvaluator 에선 output["panoptic_seg"] 를,
            #  CityscapesSemSegEvaluator 에선 output["sem_seg"] 를, 
            #  CityscapesInstanceEvaluator 에선 output["instances"] 를 이용. COCOPanopticEvaluator 에 내가 적어놓은것 참고. /21.3.29.22:20. 
            if "instances" in output:
                output = output["instances"].to(self._cpu_device)
                num_instances = len(output)
                with open(pred_txt, "w") as fout:
                    for i in range(num_instances):
                        pred_class = output.pred_classes[i]
                        classes = self._metadata.thing_classes[pred_class] # i. classes 는 카테고리명. ex: "sinus" 또는 "t_normal" 등.
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

                raise KeyError("j) 내플젝에선 이것이 프린트되지 않을거임!!!!!!!") # i. 코드조사. /21.3.28.13:50. 

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

        # groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_instanceIds.png")) # i. TODO 내플젝할때는 바로아래코드로 바꿔줘야함. /21.3.28.13:17. 
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*_instanceIds.png")) # i. 이걸 "*_labelTrainIds.png" 로 써놨었네;;;;; 아놔;;;; 지금 인스턴스레벨 이밸류에이션하는데;;; /21.3.30.23:47. 
        

        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )


        # i.21.3.28.12:12) 바로아래에서 이용하는 cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling 의 getPrediction 함수 대신 사용해줄,
        #  내가만든 getPredictionJ 함수. (똑같은 일을 하는데, 내플젝에 맞게 만든거임. 하는일은 매우 단순.)
        #  참고로, evalInstanceLevelSemanticLabeling 의 getPrediction 함수는 evalPixelLevelSemanticLabeling 의 getPrediction 과는 좀 다름.
        #  따라서, 지금 이 getPredictionJ 도 CityscapesSemSegEvaluator 의 evaluate 에서 사용해주는 getPredictionJ 와는 좀 다름. 
        def getPredictionJ(predTxtAndPngsDirpathJ, gtPngPathJ):
            gtPngFnameJ = os.path.basename(gtPngPathJ) # i. "~~_labelTrainIds.png"
            imgIdJ = gtPngFnameJ[:-len("_labelTrainIds.png")] # i. ex: "impA_BBB"
            # i. 이 리스트의 원소는 딱 1개일거임. 특정 이미지id 에 대응되는 프레딕션결과txt 파일은 1개. 
            #  참고로 프레딕션결과png파일은 각 이미지마다 해당 이미지의 인스턴스갯수만큼 생성되네. 바로위 process 보면. /21.3.28.14:07.
            predTxtPath_listJ = glob.glob(os.path.join(predTxtAndPngsDirpathJ, imgIdJ+"_pred.txt")) 
            assert len(predTxtPath_listJ)==1, "j) 원소가 1개여야하는데 뭔가 이상하네!!!" 
            predTxtPathJ = predTxtPath_listJ[0] # i. 리스트의 원소 1개일거니까 그것을 꺼내줌. /21.3.28.14:07. 
            print(f'j) predTxtPathJ 예상: /임시/폴더의/경로/impA_BBB_pred.txt') 
            print(f'j) predTxtPathJ: {predTxtPathJ}')
            return predTxtPathJ


        predictionImgList = []
        for gt in groundTruthImgList:
            # predictionImgList.append(cityscapes_eval.getPrediction(gt, cityscapes_eval.args)) # i. TODO 내플젝할때는 바로아래코드로 바꿔줘야함. /21.3.28.13:26. 
            predictionImgList.append(getPredictionJ(cityscapes_eval.args.predictionPath, gt))
        results = cityscapes_eval.evaluateImgLists(
            # i. predictionImgList 라기보단 predictionTxtList 라고해야 더 맞겠지. /21.3.31.8:39. 
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

            #  COCOPanopticEvaluator 에선 output["panoptic_seg"] 를,
            #  CityscapesSemSegEvaluator 에선 output["sem_seg"] 를, 
            #  CityscapesInstanceEvaluator 에선 output["instances"] 를 이용. COCOPanopticEvaluator 에 내가 적어놓은것 참고. /21.3.29.22:20. 
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device).numpy() # i. vTODO 이부분 조사아직못함. /21.3.27.21:30. ->이해완료. Det2 문서 모델 아웃풋 부분 참고. /21.3.29.20:57. 
            pred = 255 * np.ones(output.shape, dtype=np.uint8) # i. 1 들로 채우네. 상관없나? 어차피 모든영역 다 그려주나? vTODO output["sem_seg"] 가 뭔지 조사필요. /21.3.27.21:30.
            # i. -> 1이아니고 255로 채우는거잖아;; /21.3.29.21:16.
            for train_id, label in trainId2label.items():
                if label.ignoreInEval:
                    continue
                # i.21.3.29.23:31) 여기선 label.id 로 그려줬지만, 필요에따라 label.trainId 로 그려줘야할수도있지. 내플젝이 그렇지(내플젝에선 현재는 label.id 랑 label.trainId 가 같아서 상관없지만). 
                #  (참고로, 내가 '그려'준다는 표현을 쓰고있는데, 이미지의 각 픽셀값을 id 등의 값으로 정해주고있어서 '그린다'는 표현을 쓰고있는중임.) 
                pred[output == train_id] = label.id 
            Image.fromarray(pred).save(pred_filename)

    def evaluate(self):
        
        # print('j) CityscapesSemSegEvaluator.evaluate starts!!! ~~~~~~~~~~~')

        comm.synchronize()
        if comm.get_rank() > 0:
            return
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        # import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval # i. TODO: 내플젝할때는 바로아랫줄코드 이용해야함. /21.3.30.10:29.나중작성. 
        import cityscapesscripts.evaluation.evalPixelLevelSemanticLabelingJ as cityscapes_eval 

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
        # groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_labelIds.png")) # i. TODO 내플젝 할때는 이한줄대신 아래의 내 코드 한줄을 살려줘야함. /21.3.26.21:51. 
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



        # i.21.3.28.12:12) 밑에서 이용하는 cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling 의 getPrediction 함수 대신 사용해줄,
        #  내가만든 getPredictionJ 함수. (똑같은 일을 하는데, 내플젝에 맞게 만든거임. 하는일은 매우 단순.)
        def getPredictionJ(predPngDirpathJ, gtPngPathJ):
            gtPngFnameJ = os.path.basename(gtPngPathJ) # i. "~~_labelTrainIds.png"
            imgIdJ = gtPngFnameJ[:-len("_labelTrainIds.png")] # i. ex: "impA_BBB"
            predPngPathJ = glob.glob(os.path.join(predPngDirpathJ, imgIdJ+"_pred.png"))[0] 
            print(f'j) predPngPathJ 예상: /임시/폴더의/경로/impA_BBB_pred.png') 
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
            # i.21.3.29.0:39) -> # i.21.3.29.22:49) 잘못알던부분 수정 (pred 나 gt 나 둘다 클래스id 들을 담고있음). 
            #  predictionImgList 는 모델이 프레딕션한 클래스id (label.trainId 말고 그냥 label.id) 들이 그려져있고 VOID는 255로 그려진 
            #  png 의 경로들의 리스트임.(모델의 인퍼런스 아웃풋에서 각 이미지에해당하는 dict 의 "sem_seg" 의 정보에 따라 그려준것임.) 
            #  (인스턴스id 가 아님!!! 클래스id임!!! 지금 이거 Cityscapes"SemSeg"Evaluator 잖아!!! /21.3.30.9:12.) 
            #  groundTruthImgList 는 내플젝의경우 ~~_labelTrainIds.png 즉 클래스id(인데 train용 id (trainId)) 가 그려진 gt png 의 경로들의 리스트임. 
            #  (내플젝에선 현재는 label.trainId 랑 label.id 랑 똑같기때문에 상관없음. 
            #   만약 달랐으면, gt 로 사용할 ~~_labelIds.png 를 만들어주든지, 아니면 모델이 프레딕션한거 png 로 그려줄때 label.trainId 로 그려주면 되지.) 
            #  (참고로, 일반적으로는 클래스라는 표현이나 카테고리라는 표현이나 똑같은의미로 쓰이는데, 
            #   cityscapes 의 labels.py 에서는 'category' 가 슈퍼카테고리를 의미함. 
            #   예를들어 차,자전거 등을 모두 vehicle 이라고한다면 vehicle 이 슈퍼카테고리일건데 
            #   이걸 cityscapes 에서는 'category' 라고 한다는거지. 나중에 헷갈릴까봐 적어둠.) 
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
