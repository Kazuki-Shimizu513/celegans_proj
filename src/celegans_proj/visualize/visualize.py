from pathlib import Path
import json
import pickle
from collections import defaultdict

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cv2

import plotly.graph_objects as go

import numpy as np
import pandas as pd
from statistics import mean, stdev 

# import datetime
import torch
import torchvision
import torchvision.transforms as transforms

import argparse 
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from celegans_proj.utils.file_import import  WDDD2FileNameUtil

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")
torch._dynamo.config.suppress_errors = True


class WDDD2AnomalyDetectionResultVisualizer:
    def __init__(
        self,
        in_dir = "/mnt/c/Users/compbio/Desktop/shimizudata",
        out_dir = "/mnt/c/Users/compbio/Desktop/shimizudata",

        exp_name = "exp_example",
        dataset_name = "WDDD2_AD",
        model_names = ["PatchCore", "ReverseDistillation"],

        mode = "pseudoAnomaly",
        anomalyKinds = ['wildType', 'gridBlack', 'patchBlack', 'zoom', 'shrink', 'oneCell'],
        # mode = "RNAi",
        # anomKinds = ['', "F10E9.8", "F54E7.3",],
        alpha_metric = 2.0, # mean ± alpha * std  
    ):

        # startTime = datetime.datetime.now()
        # exp_name = f"exp_{startTime.year}{startTime.month}{startTime.day}"
        self.in_dir = Path(in_dir)
        self.out_dir = Path(out_dir)
        self.exp_name = exp_name
        self.dataset_name = dataset_name
        self.model_names = model_names
        self.mode = mode
        self.anomalyKinds = anomalyKinds
        self.alpha = alpha_metric

    def __call__(
        self,
    ):

        model_name = "PatchCore"
        for model_name in self.model_names:

            in_dir = self.in_dir.joinpath(f"{self.exp_name}/{model_name}/{self.dataset_name}")
            out_dir = self.out_dir.joinpath(f"{self.exp_name}/{model_name}/{self.dataset_name}")
            out_dir = out_dir.joinpath(f"{self.mode}_results")
            out_dir.mkdir(parents=True, exist_ok=True)


            # Get kotai_dict and kotai_score_dict
            # kotai_dict {kotai_name: [img_name1, img_name1, ... ] }
            # kotai_score_dict {kotai_name: [pred_score1, pred_score2, ... ] }
            try:
                with open(str(in_dir.joinpath("kotai_dict.pickle")), "rb") as f:
                    kotai_dict = pickle.load(f)
                with open(str(in_dir.joinpath("kotai_score_dict.pickle")), "rb") as f:
                    kotai_score_dict = pickle.load(f)
            except:
                kotai_dict, kotai_score_dict = WDDD2AnomalyDetectionResultVisualizer._build_kotai_score_dict(
                                                    in_dir,
                                                    self.anomalyKinds,
                                                )

            # wt胚の名前リストと一つにしたリストを作成
            seijou_kotai_name_list, seijou_kotai_score_list = self._get_seijou_list(kotai_score_dict)
            thresh_p, thresh_m = self._get_thresholds(seijou_kotai_score_list, self.alpha)
            logger.info(f'thresh score : {thresh_m} to {thresh_p}')


            # Plot Kotai Result
            self.violin_plot_groupBy_WT(
                kotai_dict,
                kotai_score_dict,
                seijou_kotai_name_list, seijou_kotai_score_list,
                thresh_p, thresh_m,
                out_dir,
                self.mode,
                self.anomalyKinds, 
            )
            self.line_plot_for_each_kotai(
                kotai_dict,
                kotai_score_dict,
                out_dir,
                thresh_p, thresh_m,
            )
            self.detectionScore_for_each_kotai(
                kotai_dict,
                kotai_score_dict,
                out_dir,
                thresh_p, thresh_m,
            )



    # ['gridBlack', 'patchBlack', 'zoom', 'shrink', 'oneCell']
    def violin_plot_groupBy_WT(
        self,
        kotai_dict, # kotai : img_names_list
        kotai_score_dict, # kotai_name : img_scores_list
        seijou_kotai_name_list, seijou_kotai_score_list,
        thresh_p, thresh_m,
        out_dir,
        mode="pseudoAnomaly",
        kinds = ['wildType', 'gridBlack', 'patchBlack', 'zoom', 'shrink', 'oneCell'],
    ):
        # 同じｗｔ胚，同じＲＮＡi胚ごとに分ける
        if mode=='pseudoAnomaly' :
            seijou_kotai_goto_list = self._get_pseudo_wt_list(kotai_score_dict, seijou_kotai_name_list)
        if mode=='RNAi':
            seijou_kotai_goto_list = self._get_RNAi_wt_list(kotai_score_dict, seijou_kotai_name_list, 
                                                      seijou_kotai_score_list, kinds)
        out_dir = Path(out_dir)
        out_dir_tmp = out_dir.joinpath(f"violin")
        out_dir_tmp.mkdir(exist_ok=True, parents=True)

        for onaji_seijou_kotai_dict in seijou_kotai_goto_list:
                seijou_set_and = set(seijou_kotai_name_list) & set(onaji_seijou_kotai_dict.keys())
                if len(seijou_set_and) != 1:
                    continue
                seijou_kotai_name = list(seijou_set_and)[0]

                min_score = 10000; max_score = 0
                fig = go.Figure()

                for kind in kinds:
                    if kind == 'wildType':
                        kotai_name = seijou_kotai_name
                        kind = "original"

                        scores = onaji_seijou_kotai_dict[kotai_name]
                        # get min_max score for plotting
                        if min_score > min(scores):
                            min_score = min(scores)
                        if max_score < max(scores):
                            max_score = max(scores)

                        fig.add_trace(
                            go.Violin(
                                y=scores,
                                name=kind,
                            )
                        )

                    elif mode=="pseudoAnomaly" :
                        # ['', 'gridBlack', 'patchBlack', 'zoom', 'shrink', 'oneCell']
                        pseudoAnomaly_name = WDDD2FileNameUtil.convert_name_from_wt_to_pseudoAnomaly(
                            seijou_kotai_name, kind)
                        scores = onaji_seijou_kotai_dict[pseudoAnomaly_name]
                        if len(scores) == 0:
                            logger.debug(f"{pseudoAnomaly_name}, {seijou_kotai_name}")
                            break 
                        # get min_max score for plotting
                        if min_score > min(scores):
                            min_score = min(scores)
                        if max_score < max(scores):
                            max_score = max(scores)

                        fig.add_trace(
                            go.Violin(
                                y=scores,
                                name=kind,
                            )
                        )


                    elif mode=="RNAi":
                        # TODO :: plot for each locus name
                        RNAi_names = set(onaji_seijou_kotai_dict.keys()) - seijou_set_and
                        for kotai_name in list(RNAi_names):
                            scores = onaji_seijou_kotai_dict[kotai_name]
                            # get min_max score for plotting
                            if min_score > min(scores):
                                min_score = min(scores)
                            if max_score < max(scores):
                                max_score = max(scores)

                            fig.add_trace(
                                go.Violin(
                                    y=scores,
                                    name=kotai_name,
                                )
                            )
                title =  f"{seijou_kotai_name}_{mode}"

                fig.update_traces(
                    meanline_visible=True,
                    points="all",
                    box_visible=True,
                )

                fig.add_hline(y=thresh_p, line_color="green")
                if thresh_p < max_score:
                    fig.add_hrect(y0=thresh_p, y1=max_score, line_width=0, fillcolor="green", opacity=0.1)
                fig.add_hline(y=thresh_m, line_color="green")
                if thresh_m > min_score:
                    fig.add_hrect(y0=min_score, y1=thresh_m, line_width=0, fillcolor="green", opacity=0.1)

                fig.update_yaxes(
                    title=dict(text="Anomaly Score"),
                    showline=True,
                    linewidth=1,
                    linecolor="lightgrey",
                    color="grey",
                )
                fig.update_xaxes(
                    showline=True,
                    linewidth=1,
                    linecolor="lightgrey",
                    color="grey",
                    tickfont={"size": 17},
                )
                fig.update_layout(
                    title_text=f"{title}",
                    title_x=0.5,
                    violingap=0,
                    violinmode="overlay",
                    showlegend=False,
                )
                fig.write_image(str(out_dir_tmp.joinpath(f"{title}.png" )))
                # fig.show()

    def _get_seijou_list(self,kotai_score_dict):
        # wt 胚の名前とスコアだけ抜き取る
        seijou_kotai_name_list = list()
        seijou_kotai_score_list = list()
        for kotai_name, scores in kotai_score_dict.items():
            kotai_name_elements = kotai_name.split("_")

            if "wt" not in kotai_name_elements:
                continue
            if "pseudoAnomaly" in kotai_name_elements: 
                continue #  omit like this name
            if "RNAi" in kotai_name_elements: 
                continue # RNAi_F54E7.3_100706_03 omit like this name

            seijou_kotai_name_list.append(kotai_name)
            seijou_kotai_score_list.extend(scores)
        return seijou_kotai_name_list, seijou_kotai_score_list

    def _get_pseudo_wt_list(self, kotai_score_dict, seijou_kotai_name_list):
        # 同じ正常個体から作ったpseudo異常個体をいれてまとめたい
        seijou_kotai_goto_list = list()
        for seijou_kotai_name in seijou_kotai_name_list:
            onaji_seijou_kotai_dict = defaultdict(list)
            for kotai_name, scores in kotai_score_dict.items():
                kotai_name_elements = kotai_name.split("_")
                kotai_name_elements[0] = "wt"
                kotai_name_elements[1] = "N2"
                is_seijou_kotai_name = "_".join(kotai_name_elements)
                if (
                    kotai_name != seijou_kotai_name # wt_N2_081001_01
                    and seijou_kotai_name != is_seijou_kotai_name
                ):
                    continue

                # print(f"in onaji_seijou_kotai_list {kotai_name}")
                onaji_seijou_kotai_dict[kotai_name].extend(scores)
            seijou_kotai_goto_list.append(onaji_seijou_kotai_dict)
        return seijou_kotai_goto_list


    def _get_RNAi_wt_list(self, kotai_score_dict, seijou_kotai_name_list, 
                         seijou_kotai_score_list, kinds):
        # TODO : ＲＮＡiをしゅるいごとにまとめる
        seijou_kotai_goto_list = list()
        # for seijou_kotai_name in seijou_kotai_name_list:
        seijou_kotai_name = seijou_kotai_name_list[0]
        seijou_kotai_score = seijou_kotai_score_list
        logger.debug(f"{seijou_kotai_name}, {seijou_kotai_score}")
        for kind in kinds:
            onaji_seijou_kotai_dict = defaultdict(list)
            onaji_seijou_kotai_dict[seijou_kotai_name].extend(seijou_kotai_score)
            for kotai_name, scores in kotai_score_dict.items():
                kotai_name_elements = kotai_name.split("_")
                if "RNAi" not in kotai_name_elements: 
                    continue # RNAi_F54E7.3_100706_03 omit other name
                if kind not in kotai_name_elements:
                    continue
                logger.debug(f"{kotai_name} {scores}")
                onaji_seijou_kotai_dict[kotai_name].extend(scores)
            seijou_kotai_goto_list.append(onaji_seijou_kotai_dict)
        return seijou_kotai_goto_list


    def _get_thresholds(self,scores, alpha):
        # get_thresholds
        thresh_p = mean(scores) + alpha * stdev(scores)
        thresh_m = mean(scores) - alpha * stdev(scores)
        return thresh_p, thresh_m 



    def line_plot_for_each_kotai(
        self,
        kotai_dict,
        kotai_score_dict,
        out_dir,
        thresh_p, thresh_m,
    ):
        out_dir_tmp = out_dir.joinpath(f"line")
        out_dir_tmp.mkdir(exist_ok=True, parents=True)

        for kotai_name, scores in kotai_score_dict.items():
            X = list()
            img_names = kotai_dict[kotai_name]
            for img_name in img_names:
                [kotai_kind, kotai_name, T, Z,] = WDDD2FileNameUtil.strip_name(img_name)
                X.append(f"T{str(T)}_Z{str(Z)}")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=X,
                    y=scores,
                    name=kotai_name,
                    mode="lines+markers",
                )
            )

            # fig.update_traces(points='all',)

            fig.add_hline(y=thresh_p, line_color="green")
            fig.add_hline(y=thresh_m, line_color="green")
            ranges = self._get_anomaly_ranges(scores,thresh_p,thresh_m,)
            for r in ranges:
                fig.add_vrect(
                    x0=r[0],
                    x1=r[1],
                    fillcolor="green",
                    opacity=0.25,
                )

            fig.update_yaxes(
                title=dict(text="score"),
                showline=True,
                linewidth=1,
                linecolor="lightgrey",
                color="grey",
            )
            fig.update_xaxes(
                title=dict(text="time (depth) point"),
                showline=True,
                linewidth=1,
                linecolor="lightgrey",
                color="grey",
                tickfont={"size": 24},
            )
            fig.update_layout(
                title_text=f"{kotai_name}",
                title_x=0.5,
                violingap=0,
                violinmode="overlay",
            )
            fig.write_image(str(out_dir_tmp.joinpath(f"{kotai_name}.png")))
            # fig.show()

    def _get_anomaly_ranges(
        self,
        scores,
        thresh_p,
        thresh_m,
    ):

        ranges =[]
        scores = np.array(scores)
        flags = (scores >= thresh_p) | (scores <= thresh_m)

        # 初期条件
        start = None
        end = None
        state = False
        for idx, flag in enumerate(flags):
            # 境界条件
            if idx == 0:
                start = idx
                state = True 
            if idx+1 == len(flags):
                end = idx
                state = False

            if state and not flag:#前のFlagがTrue#今回のFlagaがFalse
                end = idx
                state = False
            elif flag:##前のFlagがFalse, 今回のFlagがTrue
                start = idx
                state = True 

            if start is not None and end is not None:
                ranges.append((start,end))
                start = None
                end = None
        return ranges


    def detectionScore_for_each_kotai(
        self,
        kotai_dict,
        kotai_score_dict,
        out_dir,
        thresh_p, thresh_m,
    ):
        out_dir_tmp = out_dir.joinpath("detectionScore")
        out_dir_tmp.mkdir(exist_ok=True, parents=True)

        detection_dict = dict()
        for kotai_name, scores in kotai_score_dict.items():
            kinds_dict = defaultdict(list)
            for idx, score in enumerate(scores):
                if score >= thresh_p or score <= thresh_p:
                    kinds_dict["anomaly_idx"].append(idx)
                    kinds_dict["anomaly_score"].append(score)
                else:
                    kinds_dict["normal_idx"].append(idx)
                    kinds_dict["normal_score"].append(score)

            if "anomaly_idx" not in kinds_dict:
                kinds_dict["anomaly_idx"].append(-1)
                kinds_dict["anomaly_score"].append(-1)

            if len(kinds_dict["anomaly_idx"])>0:
                detection_rate = len(kinds_dict["anomaly_idx"]) / len(scores)
            else:
                detection_rate = 0.0

            kinds_dict["detection_rate"].append(detection_rate)
            kinds_dict["avg_anomaly_score"].append(mean(kinds_dict["anomaly_score"]))
            # kinds_dict["stdev_anomaly_score"].append(stdev(kinds_dict["anomaly_score"]))

            detection_dict[str(kotai_name)] = kinds_dict


        detection_rate_forEach_kinds = defaultdict(list)
        for kotai_name, kinds_dict in detection_dict.items():
            kind = WDDD2FileNameUtil.get_kotai_kind(kotai_name)
            detection_rate_forEach_kinds[kind].append(kinds_dict["detection_rate"][0])


        with open(str(out_dir_tmp.joinpath("detectionResults.json")), "w") as f:
            f.write(json.dumps(detection_dict, indent=4,))
            f.write(json.dumps(detection_rate_forEach_kinds, indent=4,))
     

        with open(str(out_dir_tmp.joinpath("detectionResults.txt")), "w") as f:
            for key, kinds_dict in detection_dict.items():
                f.write('%s:\n' % (key))
                for key, value in kinds_dict.items():
                    f.write('%s:%s\n' % (key, str(value)))
                f.write('\n')
            for kind, detection_rates in detection_rate_forEach_kinds.items():
                f.write(f'{kind}: {str(mean(detection_rates))}\n')
                f.write(f'{kind}: {detection_rates}\n')




    @staticmethod
    def _build_kotai_score_dict(
        in_dir ,
        anomalyKinds,
    ):
        # Build Kotai (Anomaly) Score dict
        # kotai_dict {kotai_name: [img_name1, img_name1, ... ] }
        # kotai_score_dict {kotai_name: [pred_score1, pred_score2, ... ] }
        kotai_dict = defaultdict(list)
        kotai_score_dict = defaultdict(list)
        with logging_redirect_tqdm(loggers=[logger]):
            for target_data in tqdm(anomalyKinds):
                # if target_data == "wildType":
                #     continue
                # Load predictions (list of dicts)
                predictions = torch.load(
                    str(in_dir.joinpath(f"{target_data}/predictions.pt")),
                    weights_only=False
                )
                # logger.debug(predictions[0].keys())
                # DEBUG:__main__:dict_keys([
                #    'image_path', 
                #    'label', 
                #    'image', 
                #    'mask', 
                #    'anomaly_maps', 
                #    'pred_scores', 
                #    'pred_labels', 
                #    'pred_masks', 
                #    'pred_boxes', 
                #    'box_scores', 
                #    'box_labels'
                # ])
                for pred_dict in tqdm(predictions):
                    for img_path, pred_score in zip(pred_dict['image_path'], pred_dict['pred_scores']):
                        pred_score = pred_score.tolist()
                        [base_dir, filename, suffix] = WDDD2FileNameUtil.strip_path(img_path)
                        [kotai_kind, kotai_name, T, Z,] = WDDD2FileNameUtil.strip_name(filename)

                        kotai_dict[kotai_name].append(filename)
                        kotai_score_dict[kotai_name].append(pred_score)

        logger.debug(kotai_dict.keys())
        logger.debug(kotai_score_dict.keys())
        # store them 

        with open(str(in_dir.joinpath("kotai_dict.pickle")), "wb") as f:
            pickle.dump(kotai_dict, f)
        with open(str(in_dir.joinpath("kotai_score_dict.pickle")), "wb") as f:
            pickle.dump(kotai_score_dict, f)

        return kotai_dict, kotai_score_dict


def visualize(
        in_dir = "/mnt/c/Users/compbio/Desktop/shimizudata",
        out_dir = "/mnt/c/Users/compbio/Desktop/shimizudata",
        exp_name = "exp_example",
        dataset_name = "WDDD2_AD",
        model_names = ["PatchCore", "ReverseDistillation"],
        mode = "pseudoAnomaly",
        anomalyKinds = ['wildType', 'gridBlack', 'patchBlack', 'zoom', 'shrink', 'oneCell'],
        # mode = "RNAi",
        # anomKinds = ['', "F10E9.8", "F54E7.3",],
 
):
    visualizer = WDDD2AnomalyDetectionResultVisualizer(
        in_dir ,
        out_dir ,
        exp_name ,
        dataset_name ,
        model_names ,
        mode ,
        anomalyKinds ,
    )

    visualizer()


if __name__ == "__main__":

    logging.basicConfig(filename='./logs/debug.log', filemode='w', level=logging.DEBUG)

    exp_name = "exp_20241209_predict"
    model_names = ["Patchcore", "ReverseDistillation"]
    # model_names = ["SimSID"]

    # base_dir = Path("/home/skazuki/playground/CaenorhabditisElegans_AnomalyDetection_Dataset/results/")
    base_dir = Path("/home/skazuki/skazuki/result/")
    # base_dir = Path("/mnt/c/Users/compbio/Desktop/shimizudata/")

    visualize(
        in_dir = str(base_dir),
        out_dir = str(base_dir),
        exp_name = exp_name,
        dataset_name = "WDDD2_AD",
        model_names = model_names, 
        mode = "pseudoAnomaly",
        anomalyKinds = ['wildType', 'gridBlack', 'patchBlack', 'zoom', 'shrink', 'oneCell'],
    )
