from pathlib import Path, PurePath
from pprint import pprint
import datetime
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
from statistics import mean, stdev  #

import torch
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay


from sklearn.metrics import det_curve
from sklearn.metrics import DetCurveDisplay

def plot_2img(img_npy1, img_npy2, title, title1, title2 , out_dir, fontsize=24, vmax=255, vmin=0):
    fig, ((ax00, ax01))= plt.subplots(1, 2, figsize=(8,4), tight_layout=True)
 
    ax00.imshow(img_npy1, cmap="gray", vmax=vmax, vmin=vmin)
    ax00.set_title(f'{title1}', fontsize=fontsize)
    ax00.set_axis_off()

    ax01.imshow(img_npy2, cmap="gray", vmax=vmax, vmin=vmin)
    ax01.set_title(f'{title2}', fontsize=fontsize)
    ax01.set_axis_off()

    fig.suptitle(f"{title}", fontsize=fontsize)
    plt.savefig(out_dir.joinpath(f"{title}.png"))
    plt.clf()
    plt.close()

def plot_histgram(img_npy, title, out_dir, vmax=255, vmin=0):
    # if not isinstance(img_npy, (numpy.ndarray, numpy.generic) ) and\
    #         isinstance(img_npy, (str, Path) ):
    #     img_npy = cv2.imread(str(img_npy), cv2.IMREAD_GRAYSCALE)
    # else:
    #     raise TypeError(f'please give numpy.ndarray img, you give {type(img_npy)=}')

    hist,bins = np.histogram(img_npy.flatten(),256,[vmin, vmax])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img_npy.flatten(),256,[vmin, vmax], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.title(f"{title}_histgram")
    plt.savefig(out_dir.joinpath(f"{title}_histgram.png"))
    # plt.show()
    plt.clf()
    plt.close()

def check_imgs(data_loader, out_path):
    imgs = next(iter(data_loader))
    imgs = imgs['input']
    # print(imgs,type(imgs))
    # print(imgs.shape, imgs.max(), imgs.min())
    img = torchvision.utils.make_grid(imgs)
    img = transforms.functional.to_pil_image(img)
    img.save(str(out_path)+'.png')

def calc_regular_score(scores):
    if type(scores) == torch.Tensor:
        scores = scores.detach().tolist()
    anomaly_scores = []
    for score in scores:
        try:
            # anomaly_score = 1 - (( score - min(scores)) / (max(scores) - min(scores) ))
            anomaly_score = -score
        except:
            anomaly_score = 0.0
        anomaly_scores.append(anomaly_score)
    return anomaly_scores


def calc_anomaly_score(scores):
    if type(scores) == torch.Tensor:
        scores = scores.detach().tolist()
    anomaly_scores = []
    for score in scores:
        try:
            anomaly_score = 1 - ((score - min(scores)) / (max(scores) - min(scores)))
        except:
            anomaly_score = 0.0
        anomaly_scores.append(anomaly_score)
    return anomaly_scores

    
def cvt_RGB2GRAY(img):
    im_g = list()
    if img.ndim == 4:
        for idx, im in enumerate(img):
            # print(f"{im.shape=}")
            
            im_t = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im_g.append(im_t)
        im_g = np.stack(im_g, 0) # new axis
    else:
        im_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return im_g

def strip2numpyImg(img, normalize=False,):
    if torch.is_tensor(img):
        img = img.to("cpu").detach().numpy().copy()
    if img.min() <= 0 or normalize:
        img = (img - img.min()) / (img.max() - img.min())
    if img.max() <= 1.0:
        img = img * 255
    img = img.round().astype("uint8")
    if img.ndim==4 and img.shape[1]==3:
        img = img.transpose(0,2,3,1)
    return  np.squeeze(img)

def plot_each_image(
    image,
    img_name,
    recon_img_out_dir_path,
    vmax=255,
    vmin=0,
    cmap="gray",
):
    # image = strip2numpyImg(image)
    plt.imsave(
        str(recon_img_out_dir_path.joinpath(f"{img_name}.png")),
        image,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
    )

def plot_SAG_variables(
    clean_image,
    pred_x0,
    noise_pred, 
    attention_probs,
    degraded_x0,
    img_name,
    t,
    recon_img_out_dir_path,
    fontsize=12,
    vmax=255,
    vmin=0,
):
    plt.axis("off")

    fig, ((ax00, ax01, ax02, ax03, ax04, ax05)) = plt.subplots(nrows=1, ncols=6, sharey=True, )

    ax00.imshow(clean_image, cmap="gray", vmax=vmax, vmin=vmin)
    ax00.set_title("clean_image", fontsize=fontsize)
    ax00.set_axis_off()

    ax01.imshow(pred_x0, cmap="gray", vmax=vmax, vmin=vmin)
    ax01.set_title("pred_x0", fontsize=fontsize)
    ax01.set_axis_off()

    ax02.imshow(degraded_x0, cmap="gray", vmax=vmax, vmin=vmin)
    ax02.set_title("SAGed_x0", fontsize=fontsize)
    ax02.set_axis_off()

    ax03.imshow(noise_pred, cmap="gray", vmax=vmax, vmin=vmin)
    ax03.set_title("noise_pred", fontsize=fontsize)
    ax03.set_axis_off()

    diff_fig = ax04.imshow(attention_probs, cmap=cm.viridis, vmax=vmax, vmin=vmin)
    ax04.set_title("attn_prob", fontsize=fontsize)
    ax04.set_axis_off()
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(diff_fig, cax=cax)

    extent = (0, clean_image.shape[0], 0, clean_image.shape[1])
    ax05.imshow(clean_image, cmap="gray", vmax=vmax, vmin=vmin, origin="lower", extent=extent)
    ax05.imshow(
        attention_probs,
        cmap=cm.viridis,
        vmax=vmax,
        vmin=vmin,
        alpha=0.2,
        origin="lower",
        extent=extent,
    )
    ax05.set_title("overlay", fontsize=fontsize)
    ax05.set_axis_off()

    fig.suptitle(
        f"{img_name}_noiseSteps{t}",
        fontsize=fontsize,
    )

    # rect = (0, 0, 1, 0.96)  # (left,bottom,right,top)
    rect = (0, 0, 1, 1.6)  # (left,bottom,right,top)
    fig.tight_layout(rect=rect)

    plt.savefig(
        str(recon_img_out_dir_path.joinpath(f"{img_name}_noiseSteps{t}.png")),
        bbox_inches="tight",
    )
    # plt.show()

    plt.clf()
    plt.close()


def plot_all_anomality(
    image,
    out_image,
    diff_image,
    img_name,
    recon_img_out_dir_path,
    psnr_score,
    fontsize=12,
    vmax=255,
    vmin=0,
    cmap="gray",
):
    plt.axis("off")

    fig, ((ax00, ax01, ax02, ax03)) = plt.subplots(nrows=1, ncols=4, sharey=True)

    ax00.imshow(image, cmap=cmap, vmax=255, vmin=0)
    ax00.set_title("original", fontsize=fontsize)
    ax00.set_axis_off()

    ax01.imshow(out_image, cmap=cmap, vmax=255, vmin=0)
    ax01.set_title("recon", fontsize=fontsize)
    ax01.set_axis_off()

    diff_fig = ax02.imshow(diff_image, cmap=cm.viridis, vmax=vmax, vmin=vmin)
    ax02.set_title("diffs", fontsize=fontsize)
    ax02.set_axis_off()
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(diff_fig, cax=cax)

    extent = (0, image.shape[0], 0, image.shape[1])
    ax03.imshow(image, cmap=cmap, vmax=255, vmin=0, origin="lower", extent=extent)
    ax03.imshow(
        diff_image,
        cmap=cm.viridis,
        vmax=vmax,
        vmin=vmin,
        alpha=0.2,
        origin="lower",
        extent=extent,
    )
    ax03.set_title("error map", fontsize=fontsize)
    ax03.set_axis_off()

    plt.suptitle(
        f"{img_name}\n{psnr_score=:.1f}",
        fontsize=fontsize,
    )

    rect = [0, 0, 1, 1.4]  # (left,bottom,right,top)
    plt.tight_layout(rect=rect)
    plt.savefig(
        str(recon_img_out_dir_path.joinpath(f"{img_name}.png")),
        bbox_inches="tight",
    )
    # plt.show()

    plt.clf()
    plt.close()

def get_seijou_list(kotai_dict):
    # wt 胚の名前とスコアだけ抜き取る
    seijou_kotai_name_list = list()
    seijou_kotai_score_list = list()
    for kotai_name, scores in kotai_dict.items():
        kotai_name_elements = kotai_name.split("_")
        if len(kotai_name_elements) != 4:  # wt_N2_081001_01
            continue # wt_N2_081106_02_zoom omit like this name

        if "RNAi" in kotai_name_elements: 
            continue # RNAi_F54E7.3_100706_03 omit like this name

        seijou_kotai_name_list.append(kotai_name)
        seijou_kotai_score_list.extend(scores)
    return seijou_kotai_name_list, seijou_kotai_score_list

def get_pseudo_wt_list(kotai_dict, seijou_kotai_name_list):
    # 同じ正常個体から作ったpseudo異常個体をいれてまとめたい
    seijou_kotai_goto_list = list()
    for seijou_kotai_name in seijou_kotai_name_list:
        onaji_seijou_kotai_dict = defaultdict(list)
        for kotai_name, scores in kotai_dict.items():
            kotai_name_elements = kotai_name.split("_")
            kotai_name_elements.pop() # zoom
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


def get_RNAi_wt_list(kotai_dict, seijou_kotai_name_list, 
                     seijou_kotai_score_list, kinds):
    # TODO : ＲＮＡiをしゅるいごとにまとめる
    seijou_kotai_goto_list = list()
    # for seijou_kotai_name in seijou_kotai_name_list:
    seijou_kotai_name = seijou_kotai_name_list[0]
    seijou_kotai_score = seijou_kotai_score_list
    print(f"{seijou_kotai_name}, {seijou_kotai_score}")
    for kind in kinds:
        onaji_seijou_kotai_dict = defaultdict(list)
        onaji_seijou_kotai_dict[seijou_kotai_name].extend(seijou_kotai_score)
        for kotai_name, scores in kotai_dict.items():
            kotai_name_elements = kotai_name.split("_")
            if "RNAi" not in kotai_name_elements: 
                continue # RNAi_F54E7.3_100706_03 omit other name
            if kind not in kotai_name_elements:
                continue
            print(f"{kotai_name} {scores}")
            onaji_seijou_kotai_dict[kotai_name].extend(scores)
        seijou_kotai_goto_list.append(onaji_seijou_kotai_dict)
    return seijou_kotai_goto_list



# ['gridBlack', 'patchBlack', 'zoom', 'shrink', 'oneCell']
def metric_violin_plot_for_all_kotai(
    kotai_dict,
    out_dir,
    alpha_metric=2.0,
    mode="pseudo",
    kinds = ['', 'gridBlack', 'patchBlack', 'zoom', 'shrink', 'oneCell'],
):
    # 同じｗｔ胚，同じＲＮＡi胚ごとに分けたい
    seijou_kotai_name_list, seijou_kotai_score_list = get_seijou_list(kotai_dict)
    if mode=='pseudo' :
        seijou_kotai_goto_list = get_pseudo_wt_list(kotai_dict, seijou_kotai_name_list)
    if mode=='RNAi':
        seijou_kotai_goto_list = get_RNAi_wt_list(kotai_dict, seijou_kotai_name_list, 
                                                  seijou_kotai_score_list, kinds)

    # SSIMは0のときに画質最低、1のときに画質最高
    # psnr は0～＋∞の間で、値が大きいほど画質が良いことを示す
    thresh_p = mean(seijou_kotai_score_list) + alpha_metric * stdev(
        seijou_kotai_score_list
    )
    thresh_m = mean(seijou_kotai_score_list) - alpha_metric * stdev(
        seijou_kotai_score_list
    )
    print(f'violine thresh score : {thresh_m} to {thresh_p}')

    out_dir_tmp = out_dir.joinpath(f"{mode}/violin/")
    out_dir_tmp.mkdir(exist_ok=True, parents=True)

    for onaji_seijou_kotai_dict in seijou_kotai_goto_list:
            seijou_set_and = set(seijou_kotai_name_list) & set(onaji_seijou_kotai_dict.keys())
            if len(seijou_set_and) != 1:
                continue
            seijou_kotai_name = list(seijou_set_and)[0]

            min_score = 10000; max_score = 0
            fig = go.Figure()

            for kind in kinds:
                if kind == '':
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

                elif mode=="pseudo" :
                    # ['', 'gridBlack', 'patchBlack', 'zoom', 'shrink', 'oneCell']
                    kotai_name = seijou_kotai_name + "_" + kind
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
                    title =  f"{seijou_kotai_name}_{mode}"


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
                    title = f"{seijou_kotai_name}_{kind}"




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
                title=dict(text="Anomaly Score(PSNR)"),
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
            fig.write_image(
                str(
                    out_dir_tmp.joinpath(
                       f"{title}.png" 
                    )
                )
            )
            # fig.show()


def get_kotai_anomalyScore(kotai_ssim_dict, alpha_metric=2.0):
    kotai_anomalyScore_ssim_dict = defaultdict(list)
    for kotai_name, scores in kotai_ssim_dict.items():
        anomaly_scores = calc_anomaly_score(scores)
        kotai_anomalyScore_ssim_dict[kotai_name].extend(anomaly_scores)
    seijou_kotai_anomalyScore_ssim_list = list()

    for kotai_name, scores in kotai_anomalyScore_ssim_dict.items():
        kotai_name_elements = kotai_name.split("_")
        if len(kotai_name_elements) != 4:  # wt_N2_081001_01
            continue

        seijou_kotai_anomalyScore_ssim_list.extend(scores)

    thresh_ssim = mean(seijou_kotai_anomalyScore_ssim_list) + alpha_metric * stdev(
        seijou_kotai_anomalyScore_ssim_list
    )

    return kotai_anomalyScore_ssim_dict, thresh_ssim


def get_kotai_regularScore(kotai_ssim_dict, alpha_metric=2.0):
    kotai_anomalyScore_ssim_dict = defaultdict(list)
    for kotai_name, scores in kotai_ssim_dict.items():
        anomaly_scores = calc_regular_score(scores)
        kotai_anomalyScore_ssim_dict[kotai_name].extend(anomaly_scores)

    seijou_kotai_anomalyScore_ssim_list = list()
    for kotai_name, scores in kotai_anomalyScore_ssim_dict.items():
        kotai_name_elements = kotai_name.split("_")
        if len(kotai_name_elements) != 4:  # wt_N2_081001_01
            continue
        if "RNAi" in kotai_name_elements:
            continue
        seijou_kotai_anomalyScore_ssim_list.extend(scores)

    thresh_ssim = mean(seijou_kotai_anomalyScore_ssim_list) + alpha_metric * stdev(
        seijou_kotai_anomalyScore_ssim_list
    )

    return kotai_anomalyScore_ssim_dict, thresh_ssim


def seijou_metric_line_plot_for_each_kotai(
    kotai_ssim_dict,
    test_recon_img_out_dir_path,
    alpha_metric=2.0,
    mode="ssim",
):
    kotai_ssim_dict, thresh_ssim = get_kotai_regularScore(kotai_ssim_dict, alpha_metric)

    print(f'line thresh score : {thresh_ssim}')

    test_recon_img_out_dir_path_tmp = test_recon_img_out_dir_path.joinpath(
        f"{mode}/line/seijou"
    )
    test_recon_img_out_dir_path_tmp.mkdir(exist_ok=True, parents=True)

    for kotai_name, scores in kotai_ssim_dict.items():
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                # x=time,
                y=scores,
                name=kotai_name,
                mode="lines+markers",
            )
        )

        # fig.update_traces(points='all',)

        fig.add_hline(y=thresh_ssim, line_color="green")

        start = None
        end = None
        for idx, score in enumerate(scores):
            if score >= thresh_ssim:
                if start is None:
                    start = idx
            elif start is not None:
                end = idx
                fig.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor="green",
                    opacity=0.25,
                )
                start = None
                end = None

            if score >= thresh_ssim:
                if score == scores[-1]:
                    if start is not None:
                        end = idx
                        fig.add_vrect(
                            x0=start,
                            x1=end,
                            fillcolor="green",
                            opacity=0.25,
                        )

        fig.update_yaxes(
            title=dict(text="-PSNR"),
            showline=True,
            linewidth=1,
            linecolor="lightgrey",
            color="grey",
        )
        fig.update_xaxes(
            title=dict(text="time point"),
            showline=True,
            linewidth=1,
            linecolor="lightgrey",
            color="grey",
            tickfont={"size": 24},
        )
        fig.update_layout(
            title_text=f"{mode} time series {kotai_name}",
            title_x=0.5,
            violingap=0,
            violinmode="overlay",
        )
        fig.write_image(
            str(test_recon_img_out_dir_path_tmp.joinpath(f"{kotai_name}.png"))
        )
        # fig.show()






def seijou_metric_detectionScore_for_each_kotai(
    kotai_dict,
    test_recon_img_out_dir_path,
    alpha_metric=2.0,
    mode="ssim",
):

    test_recon_img_out_dir_path_tmp = test_recon_img_out_dir_path.joinpath(
        f"{mode}/detectionScore/seijou"
    )
    test_recon_img_out_dir_path_tmp.mkdir(exist_ok=True, parents=True)

    kotai_dict, thresh = get_kotai_regularScore(kotai_dict, alpha_metric)
    detection_dict = dict()
    for kotai_name, scores in kotai_dict.items():
        kinds_dict = defaultdict(list)
        for idx, score in enumerate(scores):
            if score > thresh:
                kinds_dict["anomaly_idx"].append(idx)
                kinds_dict["anomaly_score"].append(score)
            else:
                kinds_dict["normal_idx"].append(idx)
                kinds_dict["normal_score"].append(score)

        if "anomaly_idx" not in kinds_dict:
            kinds_dict["anomaly_idx"].append(-1)
            kinds_dict["anomaly_score"].append(-1)

        if len(kinds_dict["anomaly_idx"])!=1:
            detection_rate = len(kinds_dict["anomaly_idx"]) / len(scores)
        else:
            detection_rate = 0.0

        kinds_dict["detection_rate"].append(detection_rate)
        kinds_dict["avg_anomaly_score"].append(mean(kinds_dict["anomaly_score"]))
        # kinds_dict["stdev_anomaly_score"].append(stdev(kinds_dict["anomaly_score"]))

        detection_dict[str(kotai_name)] = kinds_dict


    detection_rate_forEach_kinds = defaultdict(list)
    for kotai_name, kinds_dict in detection_dict.items():
        kotai_name_el = kotai_name.split("_")
        if len(kotai_name_el)==5: # wt_N2_081002_03_zoom
            kind = kotai_name_el[-1]
        elif kotai_name_el[0]== "RNAi":
            kind = kotai_name_el[1]
        else:
            kind = kotai_name_el[0]
        detection_rate_forEach_kinds[kind].append(kinds_dict["detection_rate"][0])


    with open(str(test_recon_img_out_dir_path_tmp.joinpath("detectionResults.json")), "w") as f:
        f.write(json.dumps(detection_dict, indent=4,))
        f.write(json.dumps(detection_rate_forEach_kinds, indent=4,))
 

    with open(str(test_recon_img_out_dir_path_tmp.joinpath("detectionResults.txt")), "w") as f:
        for key, kinds_dict in detection_dict.items():
            f.write('%s:\n' % (key))
            for key, value in kinds_dict.items():
                f.write('%s:%s\n' % (key, str(value)))
            f.write('\n')
        for kind, detection_rates in detection_rate_forEach_kinds.items():
            f.write(f'{kind}: {str(mean(detection_rates))}\n')
            f.write(f'{kind}: {detection_rates}\n')


def plot_all_seijou_picture_level_report(
    kotai_ssim_dict,
    test_recon_img_out_dir_path,
    pictures_true_labels=[
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
    ],
    all_ones_flag=True,
    alpha_metric=2.0,
    fbeta=1.0,
    mode="ssim",
):

    kotai_ssim_dict, thresh = get_kotai_regularScore(kotai_ssim_dict, alpha_metric)

    test_recon_img_out_dir_path_tmp = test_recon_img_out_dir_path.joinpath(
        f"{mode}/report/seijou"
    )
    test_recon_img_out_dir_path_tmp.mkdir(exist_ok=True, parents=True)

    all_pictureLevel_anomaly_scores = list()
    all_pictureLevel_anomaly_posnegs = list()
    if all_ones_flag:
        pictures_true_labels = list()
    for kotai_name, anomaly_scores in kotai_ssim_dict.items():
        posnegs = list()
        for score in anomaly_scores:
            if score >= thresh:
                posnegs.append(1)
            else:
                posnegs.append(0)

        # ここは大丈夫
        assert len(anomaly_scores) == len(
            posnegs
        ), f"{len(anomaly_scores)=}\n{len(posnegs)=}"

        if all_ones_flag:
            bins = list()
            kotai_name_elements = kotai_name.split("_")
            if (len(kotai_name_elements) != 4) or kotai_name_elements[0]=='RNAi': 
                # wt_N2_080930_03_randomRotate or RNAi_F54E7.3_100706_03 
                for score in anomaly_scores:
                    bins.append(1)
            else: # wt_N2_081001_01
                for score in anomaly_scores:
                    bins.append(0)

            pictures_true_labels.extend(bins)

        all_pictureLevel_anomaly_scores.extend(anomaly_scores)
        all_pictureLevel_anomaly_posnegs.extend(posnegs)

    assert len(all_pictureLevel_anomaly_scores) == len(
        all_pictureLevel_anomaly_posnegs
    ), f"{len(all_pictureLevel_anomaly_scores)=}\n{len(all_pictureLevel_anomaly_posnegs)=}"
    if all_ones_flag:
        # ここは大丈夫じゃない
        assert len(all_pictureLevel_anomaly_scores) == len(
            pictures_true_labels
        ), f"{len(all_pictureLevel_anomaly_scores)=}\n{len(pictures_true_labels)=}"

    labels = ["Regular(Positive)", "Anomaly(Negative)"]

    # Confusion Matrix
    CR_dict = classification_report(
        pictures_true_labels,
        all_pictureLevel_anomaly_posnegs,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pprint(CR_dict)
    f1_score = fbeta_score(
        pictures_true_labels,
        all_pictureLevel_anomaly_posnegs,
        # average="weighted",
        beta=fbeta,
        zero_division=0,
    )

    cm = confusion_matrix(
        pictures_true_labels,
        all_pictureLevel_anomaly_posnegs,
    )
    tn, fp, fn, tp = cm.ravel()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    title = f"all picture level\nconfusion matrix\n{f1_score=}"
    # _ = disp.ax_.set_title(title)
    plt.tight_layout()
    plt.title(title)
    plt.savefig(
        str(test_recon_img_out_dir_path_tmp.joinpath(f"{title}.png")),
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()

    # ROC curve
    # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
    fpr, tpr, _ = roc_curve(
        pictures_true_labels,
        all_pictureLevel_anomaly_scores,
        pos_label=1,
    )
    AUC = roc_auc_score(
        pictures_true_labels,
        all_pictureLevel_anomaly_scores,
        # average="weighted",
    )
    title = f"all picture level\nReceiver operating characteristic Curve\n{AUC=}"

    disp = RocCurveDisplay(
        fpr=fpr,
        tpr=tpr,
        roc_auc=AUC,
        estimator_name="Anomaly detection result",
        pos_label=1,
    )
    disp.plot(
        # plot_chance_level=True,
    )

    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])

    plt.axis("square")
    plt.title(title)
    #  plt.legend()
    plt.tight_layout()
    plt.savefig(
        str(test_recon_img_out_dir_path_tmp.joinpath(f"{title}.png")),
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()

    # PR curve
    precision, recall, _ = precision_recall_curve(
        pictures_true_labels,
        all_pictureLevel_anomaly_scores,
    )
    ap_score = average_precision_score(
        pictures_true_labels,
        all_pictureLevel_anomaly_scores,
        # average="weighted",
        # pos_label=1,
    )

    title = (
        f"all picture level\nPrecision Recall Curve\naverage_precision_score {ap_score}"
    )
    disp = PrecisionRecallDisplay(
        precision=precision,
        recall=recall,
        pos_label=1,
    )
    disp.plot(
        # plot_chance_level=True,
    )

    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])

    plt.axis("square")
    plt.title(title)
    #  plt.legend()
    plt.tight_layout()
    plt.savefig(
        str(test_recon_img_out_dir_path_tmp.joinpath(f"{title}.png")),
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()

    # det curve for determinen diffetent classifer
    fpr, fnr, _ = det_curve(
        pictures_true_labels,
        all_pictureLevel_anomaly_scores,
    )
    title = f"all picture level\nDetection Error Tradeoff"
    disp = DetCurveDisplay(
        fpr=fpr,
        fnr=fnr,
        estimator_name="Anomaly detection result",
    )
    disp.plot()
    plt.axis("square")
    plt.title(title)
    #  plt.legend()
    plt.tight_layout()
    plt.savefig(
        str(test_recon_img_out_dir_path_tmp.joinpath(f"{title}.png")),
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()


def plot_kotai_result(
    kotai_ssim_dict,
    out_dir,
    alpha_metric=2.0,
    mode="pseudo",
    pictures_true_labels=[
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
    ],
    all_ones_flag=True,
    fbeta=1.0,
    kinds = ['', 'gridBlack', 'patchBlack', 'zoom', 'shrink', 'oneCell'],
):
    metric_violin_plot_for_all_kotai(
        kotai_ssim_dict,
        out_dir,
        alpha_metric,
        mode,
        kinds, 
    )
    seijou_metric_line_plot_for_each_kotai(
        kotai_ssim_dict,
        out_dir,
        alpha_metric,
        mode,
    )
    seijou_metric_detectionScore_for_each_kotai(
        kotai_ssim_dict,
        out_dir,
        alpha_metric,
        mode,
    )

    # plot_all_seijou_picture_level_report(
    #     kotai_ssim_dict,
    #     test_recon_img_out_dir_path,
    #     pictures_true_labels,
    #     all_ones_flag,
    #     alpha_metric,
    #     fbeta,
    #     mode,
    # )

if __name__ == "__main__":

    # img = np.zeros((4, 3, 256, 256))

    # print(img.shape)
    # print(img.max())
    # print(img.min())


    # img = np.random.randn(4, 3, 256, 256)

    # print(img.shape)
    # print(img.max())
    # print(img.min())


    # img = strip2numpyImg(img)
 
    # print(img.shape)
    # print(img.max())
    # print(img.min())

    # img = cvt_RGB2GRAY(img)

    # print(img.shape)
    # print(img.max())
    # print(img.min())


    startTime = datetime.datetime.now()
    exp_name = f"exp_{startTime.year}{startTime.month}{startTime.day}"
    out_dir_path_base = Path(f"/mnt/c/Users/compbio/Desktop/shimizudata")
    out_dir_path = out_dir_path_base.joinpath(f"{exp_name}")

    # mode = "pseudo"
    # anomkinds = ['', 'gridBlack', 'patchBlack', 'zoom', 'shrink', 'oneCell']
    mode = "RNAi"
    anomkinds = ['', "F10E9.8", "F54E7.3",] # ['', "F10E9.8",#(sas-4) "F54E7.3",#(par-3)]


    img_out_dir = out_dir_path.joinpath(f"{mode}_guidanced_imgs")
    datatable_path = out_dir_path.joinpath(f"{mode}_guidanced_imgs")

    with open(str(datatable_path.joinpath("kotai_dict.pickle")), "rb") as f:
        kotai_dict = pickle.load(f)
    with open(str(datatable_path.joinpath("kotai_psnr_dict.pickle")), "rb") as f:
        kotai_psnr_dict = pickle.load(f)

    # original metric
    plot_kotai_result(
        kotai_psnr_dict,
        img_out_dir,
        alpha_metric=2.0,
        mode=mode, # "pseudo",
        all_ones_flag=True,
        fbeta=1.0,
        kinds = anomkinds,
    )

