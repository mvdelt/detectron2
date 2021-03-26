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

        print('j) COCOPanopticEvaluator.process starts!!! cococococococococococococococococo')
        print(f'j) inputs: {inputs}')      
        print(f'j) len(inputs): {len(inputs)}')      # 1  (예상대로임. 테스트시에는 뱃치사이즈가 1이라고 어디서 봤음.)
        print(f'j) outputs: {outputs}') 
        print(f'j) len(outputs): {len(outputs)}')    # 1

        from panopticapi.utils import id2rgb

        for input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()
            if segments_info is None:

                # i. 현재 내플젝(panoptic deeplab 이용한 치과파노라마 panoptic seg 플젝) 에선 이거 출력됨. /21.3.26.16:06.
                print('j) (코드조사용 출력) 모델이내뱉은 아웃풋에 segments_info 가 없음!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 

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
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing), # i. 얘도 안적혀잇음........ 엥?? /21.3.26.16:12.

                            "pred_segInfo_testJ":"segInfo_testJ",
                            "isthingJ": bool(isthing), # i. 테스트. 얜 또 적힘... 뭐지?????????? /21.3.26.16:27.
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                panoptic_img += 1

            else:
                # i. 현재 내플젝(panoptic deeplab 이용한 치과파노라마 panoptic seg 플젝) 에선 얜 출력안되고있음. /21.3.26.16:06.
                print('j) (코드조사용 출력) 모델이내뱉은 아웃풋에 segments_info 가 있음!!!!!!')

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(), # i. 이것만 안적혀있음. 뭐지?????? /21.3.26.16:11.
                        "segments_info": segments_info,

                        "pred_annotation_testJ": "this_is_test_j", # i. 확인해보니 요건 잘 적혀있음. /21.3.26.16:09.
                        "png_stringJ": out.getvalue(), # i. 테스트.
                    }
                )

    def evaluate(self):
        comm.synchronize()

        print('j) COCOPanopticEvaluator.evaluate starts!!! cococococococococococococococococo')

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

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(self._predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

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
