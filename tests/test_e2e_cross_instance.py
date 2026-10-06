# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
# ========================================================================
import os
import pickle
import unittest
import warnings

import numpy as np
import numpy.typing as npt
import torch
from scipy import optimize
from scipy.spatial import distance

warnings.filterwarnings('ignore')

from fulltilt.util import inference  # noqa


def match_centers(
    pred: npt.NDArray[float],
    gt: npt.NDArray[float],
):
    dist = distance.cdist(pred, gt)
    row_ind, col_ind = optimize.linear_sum_assignment(dist)
    return row_ind, col_ind, dist[row_ind, col_ind]


class TestFullTiltCrossInstanceE2E(unittest.TestCase):

    def setUp(self):
        """This runs before the test starts."""
        if not torch.cuda.is_available():
            self.skipTest('Skipping cross-instance E2E test: No CUDA GPU available.')

    def test_cross_instance_pipeline_regression(self):
        """End-to-end test of cross-instance inference with prototypes."""
        gt_path = 'data/test_cross_expected_output.pkl'
        if not os.path.exists(gt_path):
            self.skipTest(
                f"Skipping cross-instance E2E test: Ground truth file "
                f"'{gt_path}' not found."
            )

        print(f'\n[Cross-E2E] Loading Ground Truth from {gt_path}...')
        with open(gt_path, 'rb') as file:
            gt_data = pickle.load(file)

        print('\n[Cross-E2E] Loading Model to GPU...')
        device = 'cuda'
        model, processor_3d, _ = inference.init_model(
            config_file='configs/cfg_cryo.py',
            checkpoint_path='weights/fulltilt.pth',
            device=device,
        )

        print('[Cross-E2E] Loading Source Tomogram Data...')
        try:
            tomogram = np.load('data/test_tomogram.npy')
        except FileNotFoundError:
            self.skipTest('Skipping cross-instance E2E test: Tomogram file not found.')

        nz, ny, nx = tomogram.shape
        tomogram_whd = (nx, ny, nz)

        print('[Cross-E2E] Loading Source Tilt-Series Data...')
        try:
            source_tiltseries = np.load('data/test_tiltseries.npy')
        except FileNotFoundError:
            self.skipTest('Skipping cross-instance E2E test: Source tilt-series file not found.')

        ts_min, ts_max = source_tiltseries.min(), source_tiltseries.max()
        source_tiltseries = (source_tiltseries - ts_min) / (ts_max - ts_min) * 255

        thetas = [
            float(channel_idx * 3 - 60)
            for channel_idx in range(source_tiltseries.shape[0])
        ]

        prompts_xyz = np.array([
            [121., 70., 128.],
            [281., 62., 130.],
            [47., 341., 130.],
            [278., 294., 116.],
        ])
        prompts_label = [1, 1, 1, 1]
        diameters = np.array([10.0, 10.0, 10.0, 10.0])

        print('[Cross-E2E] Extracting Prototypes from Source Tilt-Series...')
        prototypes_dict = inference.pipeline_get_prototypes(
            model=model,
            tomogram_whd=tomogram_whd,
            tiltseries=source_tiltseries,
            thetas=thetas,
            prompts_xyz=prompts_xyz,
            prompts_label=prompts_label,
            diameters=diameters,
            device=device,
        )

        print('[Cross-E2E] Loading Target Tilt-Series Data...')
        try:
            target_tiltseries = np.load('data/test_tiltseries_cross.npy')
        except FileNotFoundError:
            self.skipTest('Skipping cross-instance E2E test: Target tilt-series file not found.')

        ts_min, ts_max = target_tiltseries.min(), target_tiltseries.max()
        target_tiltseries = (target_tiltseries - ts_min) / (ts_max - ts_min) * 255

        print('[Cross-E2E] Running Inference with Prototypes...')
        centers, estimated_diameters, labels, scores = (
            inference.pipeline_inference_with_prototypes(
                model=model,
                processor_3d=processor_3d,
                tomogram_whd=tomogram_whd,
                tiltseries=target_tiltseries,
                thetas=thetas,
                prototypes_dict=prototypes_dict,
                device=device,
            )
        )

        print('[Cross-E2E] Verifying Outputs against Ground Truth...')
        self.assertEqual(
            centers.shape,
            gt_data['centers'].shape,
            'Center shapes do not match.',
        )

        # Handle hardware differences by matching predicted centers to GT centers.
        row_ind, col_ind, dists = match_centers(centers, gt_data['centers'])
        threshold = 1.0
        inliers = dists < threshold

        np.testing.assert_allclose(
            centers[row_ind][inliers],
            gt_data['centers'][col_ind][inliers],
            atol=1.0,
            rtol=0.0,
            err_msg='Estimated centers drifted from Ground Truth.',
        )

        np.testing.assert_allclose(
            estimated_diameters[row_ind][inliers],
            gt_data['diameters'][col_ind][inliers],
            atol=1.0,
            rtol=0.0,
            err_msg='Estimated diameters drifted from Ground Truth.',
        )

        np.testing.assert_array_equal(
            labels[row_ind][inliers],
            gt_data['labels'][col_ind][inliers],
            err_msg='Estimated labels do not match Ground Truth.',
        )

        corr = np.corrcoef(
            scores[row_ind][inliers],
            gt_data['scores'][col_ind][inliers],
        )[0, 1]
        self.assertGreater(
            corr,
            0.89,
            msg='Confidence scores drifted from Ground Truth.',
        )

        outlier_ratio = (~inliers).mean()

        # NOTE: tolerate hardware / floating-point differences.
        self.assertLess(outlier_ratio, 0.32)

        print('[Cross-E2E] Success! Cross-instance pipeline matches Ground Truth.')


if __name__ == '__main__':
    unittest.main(verbosity=2)
