# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
# ========================================================================
import math

import torch
import torch.nn as nn


class TiltEncoder(nn.Module):
    def __init__(self, d_model: int = 256, max_freq_log2: float = 8.0):
        """
        Args:
            d_model: Output dimension (must match backbone feature dim)
            max_freq_log2: Max frequency power (e.g., 2^5 = 32).
                           Cryo-ET range is small (-60 to 60), we don't need super high freq.
                           2^5 means wavelength ~ 360/32 ~ 11 degrees.
                           2^7 means wavelength ~ 3 degrees.
                           Choose based on how sensitive you want it to be.
        """
        super().__init__()
        self.d_model = d_model

        self.N_freqs = d_model // 2

        self.register_buffer(
            'freq_bands',
            2.0 ** torch.linspace(0.0, max_freq_log2, self.N_freqs)
        )

        self.projection = nn.Sequential(
            nn.Linear(self.N_freqs * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, tilt_degs):
        """
        tilt_degs: [B] or [B, N] tensor of angles in degrees
        """
        t = tilt_degs.clamp(-89., 89.)

        angle = t * (math.pi / 180.0)
        args = angle.unsqueeze(-1) * self.freq_bands
        fourier_embed = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.projection(fourier_embed)
