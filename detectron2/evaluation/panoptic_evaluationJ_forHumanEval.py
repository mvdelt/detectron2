


##############################################################################################################
# i.21.4.21.19:11) panoptic_evaluation.py 를 수정해서, 모델(현재 내플젝에선 panoptic deeplab)대신
#  사람의 어노테이션 결과를 평가해주기위해 panoptic_evaluation.py 복붙해서 만든 파일. (내 논문2에 넣기위해)
#    대충다시살펴보니, DatasetEvaluator 의 process, evaluate 메서드들이 별게 아니고, 
#  process 에서 인풋과 아웃풋 짝지어주는 등 이런저런 준비해주고 evaluate 에서 본격적으로 이밸류에이션 로직이 작동하는거인듯. 
#    이거 만든 다음에, Det2 내장 panoptic deeplap 플젝의 train_net.py (detectron2_mvdeltGithub\projects\Panoptic-DeepLab\train_net.py) 
#  의 Trainer 클래스의 build_evaluator 에서 이밸류에이터들 추가해줄때 얘도 추가해주면 되겠지. 
##############################################################################################################


# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
from PIL import Image
from tabulate import tabulate

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class COCOPanopticEvaluatorJ_forHumanEval(DatasetEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(self, dataset_name, output_dir):
        """
        Args:
            dataset_name (str): name of the dataset
            output_dir (str): output directory to save results for evaluation
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }

        PathManager.mkdirs(output_dir)
        # self._predictions_json = os.path.join(output_dir, "predictions.json")
        # self._predictions_json = os.path.join(output_dir, "predictionsTestJ.json")
        self._predictions_json = os.path.join(output_dir, "predictionsTestJ_fromHuman.json") # i. /21.4.21.21:19.

    def reset(self):
        self._predictions = []

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def process(self, inputs, outputs):

        # print('j) COCOPanopticEvaluator.process starts!!! cococococococococococococococococo')
        # print(f'j) inputs: {inputs}')      
        # print(f'j) len(inputs): {len(inputs)}')      # 1  (예상대로임. 테스트시에는 뱃치사이즈가 1이라고 어디서 봤음.)
        # print(f'j) outputs: {outputs}') 
        # print(f'j) len(outputs): {len(outputs)}')    # 1

        from panopticapi.utils import id2rgb 

        for input, output in zip(inputs, outputs):
            # i.21.3.29.21:26) 
            #  여기(COCOPanopticEvaluator)서는 output["panoptic_seg"] 를 이용한다!! 
            #  CityscapesSemSegEvaluator 에선 output["sem_seg"] 를, 
            #  CityscapesInstanceEvaluator 에선 output["instances"] 를 이용함!! 
            #  Det2 문서의 모델 아웃풋 부분 보면, 모델의 아웃풋인 list[dict] 의 각 dict(하나의 이미지에 대응)가 가질수있는 키들 중 
            #  이렇게 3가지("instances", "sem_seg", "panoptic_seg") 가 주요 키들임. 
            #  Det2 내장 panoptic deeplab 플젝의 train_net.py 의 Trainer.build_evaluator 메서드 보면 
            #  이 3개의 이밸류에이터(DatasetEvaluator 클래스) 를 리스트에 담아서 DatasetEvaluator's' 로 만들어주고있지. 
            # panoptic_img, segments_info = output["panoptic_seg"] # i. 현재 내플젝(panoptic deeplab 이용한 치과파노라마 panoptic seg 플젝) 에선 segments_info 는 없음. /21.4.21.20:41. 







            # i.21.4.21.19:50) 사람이밸류에이션위해 작성한부분. output 은 모델의 아웃풋이라서 사용x. 
            #  대충 COCO 의 panopticapi 의 evaluation.py 다시보니까, 
            #  모델의 아웃풋의 한쌍(png파일과 json정보)이 서로 잘 연결되어있고,
            #  gt(ground-truth)의 한쌍(png파일과 json정보)이 서로 잘 연결되어있으면 되는거지,
            #  모델의 png파일에 기록돼있는 값과 gt의 png파일에 기록돼있는 값이 직접적으로 매치돼야하는것은 아닌듯함. 
            #  즉, 예를들어, 자전거를 모델에서는 15라는 값으로 png에 그려줬고, gt 의 png에서는 자전거의 픽셀값들이 3으로돼있다고 치면, 
            #  그렇게 값이 달라도 상관없다는얘기임. 어차피 각 값을 가지고 json정보에서 정보를 찾을거니까.
            #  근데 걍 대충 일부코드만 다시본거라 아닌부분 있을수도 있음. 

            # i. 1) input 의 이미지에 대응되는 사람의 아웃풋결과를 짝지어주기위해서, input 의 이미지파일명 추출. /21.4.21.20:59.
            file_name = os.path.basename(input["file_name"]) # i. ex) (하나의 input 에서) 'file_name': '/content/datasetsJ/panopticSeg_dentPanoJ/inputOriPano/val/imp4_188.jpg' /21.3.26.22:49.

            # i. 2) input 의 이미지에 대응되는 사람의 아웃풋결과를 찾음. /21.4.21.21:03.
            # i. 일단 테스트삼아 val용 어노테이션해준거 걍 사용해줘봄. 만약 코드가 잘 돌아간다면 평가점수가 죄다 만점이겠지.->예상대로 만점나옴. /21.4.21.20:51. 
            # i. 이제 사람의 아웃풋결과 담은 폴더 따로 만들어줬음("/content/datasetsJ/panopticSeg_dentPanoJ/gt/forHumanEval_thisIsNotGT"). /21.4.22.11:43.
            file_name_png = os.path.splitext(file_name)[0] + "_instanceIds.png" # i. ex) imp4_188_instanceIds.png /21.4.21.20:54.
            # /content/datasetsJ/panopticSeg_dentPanoJ/gt/forHumanEval_thisIsNotGT/imp4_188_instanceIds.png  
            # i. ->얘는 cityscapes 방식으로 id값들 기록된거라 모델의 출력이랑 조금 다른데(stuff 들은 1000안곱해져있고 뭐 그런식이었을거임), 걍 해보자. /21.4.21.21:01.
            # i. ->안되네 ㅋㅋ. thing 들은 죄다 만점 나오는데, stuff 는 점수 엄청 낮음. 걍 모델의 출력처럼 바꿔주자. /21.4.21.21:43.
            # i. ->현재 모델의출력이랑 동일하지는 않은상태임. 죠아래에서 stuff 의 pred_class 가 죄다 0으로 되는 문제만 해결. 아직 다른문제 발견되진 않았음. /21.4.21.21:52쯤.
            png_fromHumanJ = os.path.join("/content/datasetsJ/panopticSeg_dentPanoJ/gt/forHumanEval_thisIsNotGT", file_name_png) 
            png_arr_fromHumanJ = np.array(Image.open(png_fromHumanJ))







            # panoptic_img = panoptic_img.cpu().numpy()
            panoptic_img = png_arr_fromHumanJ
            segments_info = None # i. /21.4.21.21:11.
            if segments_info is None:

                # i. 현재 내플젝(panoptic deeplab 이용한 치과파노라마 panoptic seg 플젝) 에선 아래의 print 가 출력됨. /21.3.26.16:06.
                # print('j) (코드조사용 출력) 모델이내뱉은 아웃풋에 segments_info 가 없음!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
                #
                # i.21.3.26.21:42) ->즉, 바로아래의 Det2 기존 코멘트 설명대로일거임 아마도. 
                #  즉, 아래에서의 panoptic_label 변수(이게 모델(panoptic deeplab)이 출력한 정보인거지)는
                #  그냥 단순히 category_id * label_divisor + instance_id 로 계산된것인듯.
                #  그래서, self._predictions_json json파일(현재 내 구글드라이브에 저장되게해놨지) 열어보면
                #  cityscapes 의 방식이랑 좀 다른것을 볼수있음. 

                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        # VOID region.
                        continue




                    # pred_class = panoptic_label // label_divisor  # i. <-기존코드.
                    # i.21.4.21.21:52) 모델의 출력은 죄다 1000 이상인데, 지금 내가 사람결과 평가위해 이용해주려는건 cityscapes 형식대로라서 stuff 들은 값이 1000보다 작음. 이거해결위한코드. 
                    #    물론 이렇게해준다해서 모델의출력과 동일해지는건 아님. 
                    #  모델의 출력은 stuff 든 thing 이든 상관없이 걍 카테고리id 에다가 1000 곱한다음, thing 이면 1,2,3,... 이런식으로 1부터 더해줌. (0부터 더하는게 아니고).
                    #  반면, cityscapes 형식에서는, stuff는 카테고리id에 1000 안곱하고, thing 은 1000 곱한다음 0,1,2,... 이런식으로 0부터 더해줌. 
                    #    즉, 굳이 모델의출력과 동일하게 만들려면, cityscapes 방식의 png 파일에서,
                    #  stuff 의 (픽셀에 기록된)값에다가 label_divisor 값 (현재 1000) 곱해주고,
                    #  thing 의 (픽셀에 기록된)값에다가 1 더해주면 됨. 
                    #    근데 그렇게 하려면 할수있는데 지금 걍 이렇게만 해줘봄. ->일단 이렇게만 해도 이밸류에이션 돌려본 결과는 stuff, thing 모두 만점임(예상대로). 
                    #  TODO: 나중에 바로위에내가적은것처럼 모델의출력이랑 동일하게해서도 해보자. 혹시 결과 다른가. 
                    if panoptic_label < label_divisor:
                        pred_class = panoptic_label
                    else: 
                        pred_class = panoptic_label // label_divisor





                    isthing = (
                        pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,  # i. TODO 내플젝할때는 +1 제거해야함 /21.3.27.14:21.  # i. ->다시생각해보니, 걍 +1을 해주든 +123을 해주든 상관없이 똑같을듯..?? 아직제대로생각완료못함. /21.3.28.12:59.
                            # "id": int(panoptic_label),        # i. 내플젝 위해 +1 제거. /21.3.27.14:19. 
                            "category_id": int(pred_class),
                            "isthing": bool(isthing), # i. 얘도 안적혀잇음...엥?? /21.3.26.16:12. ->_convert_category_id 에서 pop해주자나;;; /21.3.26.18:40.
                        }
                    )
                # Official evaluation script uses 0 for VOID label.  # i. TODO 음.. 기존Det2코멘트가 틀린것같은데.. +1 해줄필요없을것같은데??? 내생각맞나테스트해보자. /21.3.28.13:09. 
                panoptic_img += 1  
                # i. TODO ->내플젝에선 이거(panoptic_img += 1) 코멘트아웃해주면 될듯.  # i. ->다시생각해보니, 걍 +1을 해주든 +123을 해주든 상관없이 똑같을듯..?? 아직제대로생각완료못함. /21.3.28.12:59. 
                #  내플젝에선 백그라운드도 하나의 foreground 처럼 프레딕션해주고있는데(시각화를위해),
                #  그 백그라운드의 id가 0이니까. self._predictions_json json파일(현재 내 구글드라이브에 저장되게해놨지) 열어보면
                #  모델(panoptic deeplab)이 프레딕션한 결과가 어떻게 출력되었나 확인가능. /21.3.26.20:22.
                # i.21.3.26.23:03) 참고로, 혹시나중에 까먹을까봐 적으면, 
                #  보통은 panoptic seg 태스크에서 백그라운드는 프레딕션시 클래스 선택지에서 없게해줌. 그게 평가방식상 점수 유리하니까. coco 나 cityscapes 의 panoptic seg 평가시.
                #  그래서 시각화시키면 백그라운드가 없음. 백그라운드도 전부 foreground 클래스들이 죄다 덮어버림(그래도 점수 안깎임).
                #  근데 나는 시각화출력시에 백그라운드도 보여줘야 예쁘니까 백그라운드도 foreground 클래스중 하나로 해준거고.

            # else:
                # i. 현재 내플젝(panoptic deeplab 이용한 치과파노라마 panoptic seg 플젝) 에선 얜 출력안되고있음. /21.3.26.16:06.
                # print('j) (코드조사용 출력) 모델이내뱉은 아웃풋에 segments_info 가 있음!!!!!!')

            file_name = os.path.basename(input["file_name"]) # i. ex) (하나의 input 에서) 'file_name': '/content/datasetsJ/panopticSeg_dentPanoJ/inputOriPano/val/imp4_188.jpg' /21.3.26.22:49.
            file_name_png = os.path.splitext(file_name)[0] + ".png" # i. ex) imp4_188.png 이런식으로 되겠지. /21.3.26.22:49.
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]

                # print(f'j) io.BytesIO() as out, out.getvalue(): {out.getvalue()}')   # b'\x89PNG\r\n\x1a\n\x00\~~~~~~~~~~'  

                self._predictions.append(
                    {
                        "image_id": input["image_id"], # i. ex) 'imp4_188' /21.3.26.22:52.
                        "file_name": file_name_png, # i. ex) imp4_188.png /21.3.26.22:49.
                        # i. 이것만 출력안됨. 뭐지?????? /21.3.26.16:11. 
                        #  ->죠아래에서 pop 해주잖아;; 이거 pop 안해주면 json.dumps 안됨(TypeError: Object of type bytes is not JSON serializable). /21.3.26.18:40.
                        "png_string": out.getvalue(), 
                        "segments_info": segments_info,
                    }
                )

    def evaluate(self):
        comm.synchronize()

        # print('j) COCOPanopticEvaluator.evaluate starts!!! cococococococococococococococococo')

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json) # i. COCO형식으로 변환된 어노json파일 경로./21.3.10.12:02.에적어뒀던것. /21.3.26.13:26. 
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root) # i. COCO형식으로 변환된 어노png파일들 있는 디렉토리./21.3.10.12:02.에적어뒀던것. /21.3.26.13:26. 

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions
            with PathManager.open(self._predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from panopticapi.evaluation import pq_compute

            # with contextlib.redirect_stdout(io.StringIO()):
            # i.21.3.27.18:34) print 출력좀 확인하려고 내가 좀 바꿔줌.
            ioJ = io.StringIO()
            with contextlib.redirect_stdout(ioJ):
                pq_res = pq_compute(
                    gt_json, # i. COCO형식으로 변환된 어노json파일 경로. /21.3.26.22:41.
                    PathManager.get_local_path(self._predictions_json), # i. 모델(현재 panoptic deeplab)의 출력 json 경로.(위에서 gt_json 의 "annotations"를 바꿔서 만들어줬지.) /21.3.26.22:42.
                    gt_folder=gt_folder, # i. COCO형식으로 변환된 어노png파일들 있는 디렉토리. /21.3.26.22:55.
                    pred_folder=pred_dir, # i. 모델이 출력한 png 들을 넣어준 디렉토리 경로. (현재 임시디렉토리로 되어있지. 참고로 위에서 각 픽셀값들에다 +1해줬지. 내플젝에선 +1필요없을듯하지만.) /21.3.26.23:06. 
                )
            print(f'j) got stdout: \n{ioJ.getvalue()}') 

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        # results = OrderedDict({"panoptic_seg": res}) 
        results = OrderedDict({"panoptic_seg_humanEvalJ": res}) 
        _print_panoptic_results(pq_res)

        return results


def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    # logger.info("Panoptic Evaluation Results:\n" + table)
    logger.info("j) Panoptic Evaluation Results, for Human!!!:\n" + table) 


if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json")
    parser.add_argument("--gt-dir")
    parser.add_argument("--pred-json")
    parser.add_argument("--pred-dir")
    args = parser.parse_args()

    from panopticapi.evaluation import pq_compute

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            args.gt_json, args.pred_json, gt_folder=args.gt_dir, pred_folder=args.pred_dir
        )
        _print_panoptic_results(pq_res)
