#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Panoptic-DeepLab Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""

import os
import torch

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader,    DatasetCatalog 
# i.21.3.16.12:32) ->데이타셋레지스터도 여기(지금 이 train_net.py)서 해주려고 DatasetCatalog 도 임포트했음.
#  코랩에서 from detectron2.data.datasets.J_cityscapes_panoptic import register_all_cityscapes_panoptic 으로
#  내가수정해준 register_all_cityscapes_panoptic 함수를 임포트해서 데이터셋 레지스터해줘도, 
#  코랩에서 그 이후 셀에서 !python train_net.py 실행시키면 데이터셋이 레지스터 안된걸로 나옴.
#  즉, !python train_net.py 로 실행시킨거는 별개인거지. 
#  코랩에서 !python train_net.py 를 해준다는거는, 쉽게말해 코랩컴의 커맨드창에서 python train_net.py 를 실행시켜주는셈이니까.
#  즉, 코랩에서 셀들을 돌리는거는 코랩컴에서 어떠한 파이썬 코드를 실행하고있는건데, 추가적으로 커맨트창을 따로 열어서 python train_net.py 를 실행시켜주면,
#  이거는 실행하던 파이썬코드랑은 별개로 실행되나봄. 너무 당연한건가??!!!
#  음 그러니까 내생각엔, 아예 다른 프로세스에서 돌아가는거라고 봐야할듯. 예를들어 train_net.py 를 두번 실행시키면, 각각 별도의 프로세스에서 동작할거잖아. 뭐 그런비슷한거지.

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    CityscapesSemSegEvaluatorJ_forHumanEval, # i.21.4.22.10:10) 추가. 
    COCOEvaluator,
    COCOPanopticEvaluator,
    COCOPanopticEvaluatorJ_forHumanEval, # i.21.4.21.21:22) 추가. 
    DatasetEvaluators,
)
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.projects.panoptic_deeplab import (
    PanopticDeeplabDatasetMapper,
    add_panoptic_deeplab_config,
)
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping


# i.21.2.13.15:20) 내가추가.
from detectron2.engine import hooks


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    augs.append(T.RandomFlip())
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED:
            return None
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference") # i. 요기서 "inference" 폴더가 등장하는거였군. /21.3.25.11:58.
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["cityscapes_panoptic_seg", "coco_panoptic_seg"]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder)) ########################################### /21.3.25.11:56.
            # evaluator_list.append(COCOPanopticEvaluatorJ_forHumanEval(dataset_name, output_folder)) ############## 사람의결과 이밸류에이션 위해 추가. /21.4.21.21:21. 
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name)) ########################################### /21.3.25.11:56.
            # evaluator_list.append(CityscapesSemSegEvaluatorJ_forHumanEval(dataset_name)) ############## 사람의결과 이밸류에이션 위해 추가. /21.4.22.10:06. 
            evaluator_list.append(CityscapesInstanceEvaluator(dataset_name)) ########################################### /21.3.25.11:56.
            # i. ->사람은 스코어 내놓지 않으니까 AP 계산할수없어서 CityscapesInstanceEvaluator 는 따로 사람의결과 이밸류에이션 위해 수정해서 사용해주지 않았음. 
        if evaluator_type == "coco_panoptic_seg":
            # `thing_classes` in COCO panoptic metadata includes both thing and
            # stuff classes for visualization. COCOEvaluator requires metadata
            # which only contains thing classes, thus we map the name of
            # panoptic datasets to their corresponding instance datasets.
            dataset_name_mapper = {
                "coco_2017_val_panoptic": "coco_2017_val",
                "coco_2017_val_100_panoptic": "coco_2017_val_100",
            }
            evaluator_list.append(
                COCOEvaluator(dataset_name_mapper[dataset_name], output_dir=output_folder) # i. <- 요놈은 사용해줄까 말까? COCOPanopticEvaluator 랑 뭐가다른거지?? /21.3.25.11:56.
            )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg): # i.21.3.25.12:41) TODO Q: 왜 build_test_loader 는 오버라이드 안해주지???????????? 테스트할때도 mapper 필요하지않나??????????????
        mapper = PanopticDeeplabDatasetMapper(cfg, augmentations=build_sem_seg_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")


    # i.21.2.13.15:11) 트레이닝끝나고 이밸류에이션 안하게 해주려고(이밸류에이션 시간 넘 오래걸려서),
    #  DefaultTrainer 의 build_hooks 함수 override 한다음 훅 리스트에 hooks.EvalHook 추가하는부분 삭제.
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        
        
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # i.21.2.13.15:22)
        # print('j) This is build_hooks method, evaluation hook removed version!!')
        # # Do evaluation after checkpointer, because then if it fails,
        # # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results)) 
        # i.21.2.13.15:15) ->이부분 코멘트아웃.
        # i.21.3.13.0:36) ->다시 복구. 
        #  이밸류에이션 따로 해주면될줄알앗는데, 이거(hooks.EvalHook 관련) 생각보다 좀 복잡하네?
        #  test 함수 돌리는거 자체는 뭐 걍 하면 되는것같은데.. 왜케복잡하지.. 
        #  암튼 일단 다시 살려서 이밸류에이션 되게 해보자. 내플젝 이밸류에이션 되나 보기도 해야하니까.
        #  (사실 이렇게 살려놓을거면 DefaultTrainer 의 build_hooks 함수랑 똑같을테니
        #   이렇게 overriding 할필요 없지만, 아무튼 일단은.)
        # i.21.3.16.14:56) ->다시 코멘트아웃. 지금 내 커스텀 데이터셋에 val 준비안돼잇어서.
        # i.21.3.19.13:16) -> hooks.EvalHook 관련 EventStorage 등 조사완료. 다시보니 뭐 그닥 복잡할것도 없네.

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)


    # i.21.3.16.12:53) 이쯤에서 내 커스텀데이타셋 레지스터 해줘보자!! ##########################################
    #  (코랩의 셀에서 레지스터해주면 적용안되는이유 저위에 적어놨음)
    #  TODO: 이 코드부분 위치 옮겨야할수도 있음!! 아직 꼼꼼히 안살펴봄!!
    from detectron2.data.datasets.J_cityscapes_panoptic import register_all_cityscapes_panoptic
    dataRootJ = "/content/datasetsJ" # i. 코랩컴에서의 경로임.
    register_all_cityscapes_panoptic(dataRootJ)
    # 그리고 이제 밑에서 trainer 객체만들어서 돌려주니까 지금 이렇게 여기서 레지스터해주면 될듯..?
    ##########################################################################################################


    # i.21.3.19.14:24) hooks.EvalHook 이용해서 이밸류에이션 해주는거나 이거나 내내 똑같은거임.
    #  hooks.EvalHook 이용하면 이밸류에이션 결과(Trainer.test(~~)의 리턴값)를 EventStorage 에 저장해주고
    #  hooks.PeriodicWriter 가 그걸 출력해주는 방식이고,
    #  이거는 곧바로 Trainer.test(~~)의 리턴값을 내뱉는 방식인거고.
    # i.21.3.24.20:05) TODO Q: 근데 Trainer.test 함수를 실행만 시켜도 결과 출력되나? 
    #  여기선 리턴값을 따로 출력해주진 않는데;; 
    #  만약 실행만 시켜도 결과 출력되면, EvalHook 이용한건 똑같은게 두번출력되겟는데?
    #  근데 내기억엔 안그랬던것같은데;; 지금코드살펴볼시간없으니 일단 실행시켜보자.
    if args.eval_only:
        model = Trainer.build_model(cfg)
        # i.21.3.19.14:22) 뜯어보진않앗는데, 아마 model 에 웨잇을 로드해주기 위함인듯.
        #  (바로 위에서 실행해준 Trainer.build_model(cfg) 는 cfg 로부터 웨잇 로드 안함.)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        
        print('j) ----------------------------------------------------------')
        print(f'j) test result: {res}') # i. <-이런식으로 출력해줘보자. 똑같은거 또나오나. /21.3.24.20:18.
        
        return res



    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume) 
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
