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


class COCOPanopticEvaluator(DatasetEvaluator):
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
        self._predictions_json = os.path.join(output_dir, "predictionsTestJ.json")

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
            panoptic_img, segments_info = output["panoptic_seg"] 
            panoptic_img = panoptic_img.cpu().numpy()
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

        results = OrderedDict({"panoptic_seg": res})
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
    logger.info("Panoptic Evaluation Results:\n" + table)


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
