# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
# ========================================================================
import warnings

warnings.filterwarnings('ignore')

import numpy as np  # noqa

from fulltilt.util import inference  # noqa


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Initialization
    # ---------------------------------------------------------
    device = 'cuda'

    print('Initializing model...')
    model, processor_3d, _ = inference.init_model(
        config_file='configs/cfg_cryo.py',
        checkpoint_path='weights/fulltilt.pth',
        device=device,
    )

    # ---------------------------------------------------------
    # 2. Data Loading & Preprocessing
    # ---------------------------------------------------------
    # Define the 3D dimensions of the reconstructed tomogram (Width, Height, Depth)
    tomogram = np.load('data/test_tomogram.npy')
    nz, ny, nx = tomogram.shape
    tomogram_whd = (nx, ny, nz)

    # Load the aligned tilt-series images
    # Shape should be (Channels/Tilts, Height, Width)
    tiltseries = np.load('data/test_tiltseries.npy')

    # Normalize pixel intensities to the range [0, 255] for image processing
    ts_min, ts_max = tiltseries.min(), tiltseries.max()
    tiltseries = (tiltseries - ts_min) / (ts_max - ts_min) * 255

    # Define the angle (in degrees) for each image in the tilt-series.
    thetas = [channel_idx * 3 - 60 for channel_idx in range(tiltseries.shape[0])]

    # ---------------------------------------------------------
    # 3. Prompt Definitions
    # ---------------------------------------------------------
    # Specify initial 3D seed coordinates (prompts) in [X, Y, Z] format
    prompts_xyz = np.array(
        [
            [121., 70., 128.],
            [281., 62., 130.],
            [47., 341., 130.],
            [278., 294., 116.],
        ],
    )

    # Assign a class label to each 3D prompt
    prompts_label = [1, 1, 1, 1]

    # Provide diameter (in pixels) for the target objects
    diameters = np.array([10, 10, 10, 10])

    # ---------------------------------------------------------
    # 4. Run Inference Pipeline
    # ---------------------------------------------------------
    print('Running inference pipeline...')
    centers, diameters, labels, scores = inference.pipeline(
        model=model,
        processor_3d=processor_3d,
        tomogram_whd=tomogram_whd,
        tiltseries=tiltseries,
        thetas=thetas,
        prompts_xyz=prompts_xyz,
        prompts_label=prompts_label,
        diameters=diameters,
        device=device,
        # nms_threshold=0.5,  # NOTE: nms_threshold can be set
        # confidence_score_threshold=0.5,  # NOTE: confidence_score_threshold can be set
    )

    # ---------------------------------------------------------
    # 5. Output Results
    # ---------------------------------------------------------
    print(f'Number of detected objects: {len(centers)}')
    print('\n--- Top 10 Predictions ---')
    print(f'Centers: {centers[:10]}')
    print(f'Diameters: {diameters[:10]}')
    print(f'Labels: {labels[:10]}')
    print(f'Confidence scores: {scores[:10]}')


if __name__ == '__main__':
    main()
