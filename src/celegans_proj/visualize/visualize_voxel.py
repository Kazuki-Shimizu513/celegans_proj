# import data
import time
import numpy as np

import plotly.graph_objects as go
from scipy.spatial import Delaunay
import moviepy.editor as mpy
import io
from skimage import io as skio
from PIL import Image


def plotly_fig2array(fig):
    # convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


# to make gif from ploty, please go to below link
# https://community.plotly.com/t/how-to-export-animation-and-save-it-in-a-video-format-like-mp4-mpeg-or/64621/2


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }


def ploty3Dslice(volume, out_path):
    nb_frames, r, c = volume.shape
    # print(f"{out_path=}\t{volume.shape=}")

    z_top = nb_frames - 1
    z_top_m = z_top * 0.1
    fig = go.Figure(
        frames=[
            go.Frame(
                data=go.Surface(
                    z=(z_top_m - k * 0.1) * np.ones((r, c)),
                    surfacecolor=np.flipud(volume[z_top - k]),
                    cmin=0,
                    cmax=255,
                ),
                name=str(
                    k
                ),  # you need to name the frame for the animation to behave properly
            )
            for k in range(nb_frames)
        ]
    )

    # Add data to be displayed before animation starts
    fig.add_trace(
        go.Surface(
            z=z_top_m * np.ones((r, c)),
            surfacecolor=np.flipud(volume[z_top]),
            colorscale="Gray",
            cmin=0,
            cmax=200,
            colorbar=dict(thickness=20, ticklen=4),
        )
    )

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title="Slices in volumetric data",
        width=1200,
        height=1200,
        scene=dict(
            zaxis=dict(range=[-0.1, (nb_frames * 0.1)], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )

    fig.show()
    fig.write_html(out_path)


if __name__ == "__main__":
    vol = skio.imread(
        "https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif"
    )
    volume = vol.T
    out_path = "/mnt/f/test.html"

    ploty3Dslice(volume, out_path)
