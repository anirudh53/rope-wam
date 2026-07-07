"""
rope_wam.py  —  ROPE for WAM  
----------------------------------------------------------------
IC always comes from the IC table + driver CSV:

    rope = ROPE_WAM()
    out  = rope.run("2020-03-05 06:00:00", horizon=120)

Fusion is always meta-model based.

True density for evaluation:

    true_dens = rope.get_true_density("2020-03-05 06:00:00", horizon=120)
    # returns (H, 72, 36, 45) float32

Output keys: window_df, mean_density, density_std (if uncertainty=True)
Driver set:  f10, f10_41day_avg, ap, t1, t2, t3, t4
"""

from __future__ import annotations

import glob
import os
import yaml
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.interpolate import griddata

import tensorflow as tf
from tensorflow import keras

from ts_utils.custom_layers import PositionalEncoding
from ae_utils import utils as utils_cae
from ae_utils.attn_models import COAE

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.get_logger().setLevel("ERROR")


# ── Driver columns (fixed for WAM) ────────────────────────────────────────────
WAM_DRIVER_COLS: List[str] = ["f10", "f10_41day_avg", "ap", "t1", "t2", "t3", "t4"]


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class WAMROPEConfig:
    latent_dim:        int = 10
    seq_len:           int = 3
    decode_batch_size: int = 4

    @property
    def total_dim(self) -> int:
        return self.latent_dim + len(WAM_DRIVER_COLS)   # 17


# ============================================================
# Small utils
# ============================================================
def _to_numpy(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.array(x)


def _safe_device(device: str) -> str:
    return device if not (device.startswith("cuda") and not torch.cuda.is_available()) else "cpu"


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# Feature normalizer
# ============================================================
class FeatureNormalizer:
    def __init__(self, stats_ts: Dict[str, Any], latent_dim: int = 10):
        self.K     = latent_dim
        self.mu    = _to_numpy(stats_ts["mu"]).astype(np.float32)
        self.sigma = _to_numpy(stats_ts["sigma"]).astype(np.float32)

        if len(self.mu) != len(self.sigma):
            raise ValueError(f"stats_ts mu/sigma length mismatch: {len(self.mu)} vs {len(self.sigma)}")

        self.total_dim  = len(self.mu)
        self.driver_dim = self.total_dim - self.K

        if self.driver_dim != len(WAM_DRIVER_COLS):
            raise ValueError(
                f"stats_ts driver_dim={self.driver_dim} but WAM expects {len(WAM_DRIVER_COLS)}. "
                f"Check you loaded the correct stats_ts file."
            )

    def norm_full(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sigma).astype(np.float32)

    def norm_driver(self, drv: np.ndarray) -> np.ndarray:
        return ((drv - self.mu[self.K:]) / self.sigma[self.K:]).astype(np.float32)

    def denorm_latents(self, lat_norm: np.ndarray) -> np.ndarray:
        return (lat_norm * self.sigma[: self.K] + self.mu[: self.K]).astype(np.float32)



# ============================================================
# IC-table interpolator
# ============================================================
class ICTableInterpolator:
    def __init__(self, ic_table: pd.DataFrame):
        self.pts  = ic_table[["F10", "Ap"]].values
        self.vals = ic_table.drop(columns=["F10", "Ap"]).values

    def get_latent_coeffs(self, f10: float, ap: float) -> np.ndarray:
        query = np.array([[f10, ap]])
        pred  = griddata(self.pts, self.vals, query, method="linear")
        if np.isnan(pred).any():
            pred = griddata(self.pts, self.vals, query, method="nearest")
        return pred.flatten().astype(np.float32)


# ============================================================
# Driver window builder  (IC-table mode)
# ============================================================
class DriverWindowBuilder:
    def build(
        self,
        driver_df:      pd.DataFrame,
        start_datetime,
        horizon:        int,
        seq_len:        int,
    ) -> pd.DataFrame:
        start_dt   = pd.Timestamp(start_datetime)
        hist_start = start_dt - timedelta(hours=(seq_len - 1))
        end_dt     = start_dt + timedelta(hours=(horizon - 1))
        timeline   = pd.date_range(hist_start, end_dt, freq="h")

        df = (
            driver_df.copy()
            .assign(datetime=lambda d: pd.to_datetime(d["datetime"]))
            .set_index("datetime")
            .reindex(timeline)
            .reset_index()
            .rename(columns={"index": "datetime"})
        )

        hour       = df["datetime"].dt.hour
        doy        = df["datetime"].dt.dayofyear.astype(float)
        df["t1"]   = np.sin(2 * np.pi * hour / 24)
        df["t2"]   = np.cos(2 * np.pi * hour / 24)
        df["t3"]   = np.sin(2 * np.pi * doy / 365.25)
        df["t4"]   = np.cos(2 * np.pi * doy / 365.25)

        # 41-day rolling avg from full driver history
        drv_indexed = (
            driver_df.assign(datetime=lambda d: pd.to_datetime(d["datetime"]))
            .set_index("datetime").sort_index()
        )
        df["f10_41day_avg"] = [
            float(drv_indexed.loc[:dt, "f10"].tail(984).mean())
            for dt in timeline
        ]

        missing = df[["f10", "ap"]].isna().any(axis=1)
        if missing.any():
            bad = df.loc[missing, ["datetime", "f10", "ap"]].head(10)
            raise ValueError(f"Driver CSV has gaps near {start_datetime}.\n{bad}")

        return df


# ============================================================
# Sequence builder  (IC-table mode)
# ============================================================
class SequenceBuilder:
    def __init__(
        self,
        normalizer: FeatureNormalizer,
        ic_interp:  ICTableInterpolator,
        latent_dim: int = 10,
        seq_len:    int = 3,
    ):
        self.norm = normalizer
        self.ic   = ic_interp
        self.K    = latent_dim
        self.S    = seq_len
        self.D    = normalizer.total_dim

    def build_X_init_norm(self, window_df: pd.DataFrame) -> np.ndarray:
        coeff_rows, drv_rows = [], []
        for i in range(self.S):
            row = window_df.iloc[i]
            coeff_rows.append(self.ic.get_latent_coeffs(row["f10"], row["ap"]))
            drv_rows.append(row[WAM_DRIVER_COLS].to_numpy(dtype=np.float32))
        X = np.hstack([np.vstack(coeff_rows), np.vstack(drv_rows)])
        return self.norm.norm_full(X)

    def build_x_chunk(
        self, X_init_norm: np.ndarray, forecast_df: pd.DataFrame, horizon: int
    ) -> np.ndarray:
        x_chunk    = np.zeros((horizon, self.S, self.D), dtype=np.float32)
        x_chunk[0] = X_init_norm
        raw_drv    = forecast_df[WAM_DRIVER_COLS].to_numpy(dtype=np.float32)
        for t in range(1, horizon):
            drv_norm     = self.norm.norm_driver(raw_drv[t])
            row          = np.zeros(self.D, dtype=np.float32)
            row[self.K:] = drv_norm
            x_chunk[t]   = np.vstack([x_chunk[t - 1][1:], row])
        return x_chunk


# ============================================================
# Keras helpers
# ============================================================
def _load_keras_ensemble(
    model_dir:      str,
    n_models:       int  = 5,
    custom_objects: Optional[dict] = None,
) -> List[keras.Model]:
    return [
        keras.models.load_model(
            os.path.join(model_dir, f"best_model_{i}.keras"),
            compile=False,
            custom_objects=custom_objects,
        )
        for i in range(1, n_models + 1)
    ]


def _make_infer_fn(model: keras.Model):
    @tf.function(reduce_retracing=True)
    def infer(x):
        return model(x, training=False)
    return infer


# ============================================================
# Dynamic rollout
# ============================================================
class DynamicRollout:
    def __init__(self, latent_dim: int = 10):
        self.K = latent_dim

    def run(self, infer_fn, x_chunk_np: np.ndarray, horizon: int) -> np.ndarray:
        inp   = x_chunk_np[:1].copy()
        preds = np.zeros((horizon - 1, self.K), dtype=np.float32)
        for t in range(1, horizon):
            p            = infer_fn(tf.constant(inp)).numpy()
            preds[t - 1] = p[0].astype(np.float32)
            if (t + 1) < horizon:
                inp[0, :-1]          = inp[0, 1:]
                inp[0, -1, : self.K] = p[0]
                inp[0, -1, self.K:]  = x_chunk_np[t + 1, -1, self.K:]
        return preds


# ============================================================
# Meta fusion
# ============================================================
class MetaFusion:
    def __init__(self, coeff_level: bool = True):
        self.coeff_level = coeff_level

    def fuse(
        self,
        meta_infer_fn,
        x_chunk_np: np.ndarray,   # (T, S, D)
        all_preds:  np.ndarray,   # (M, T, K)
    ) -> np.ndarray:              # (T, K)
        W         = meta_infer_fn(tf.constant(x_chunk_np)).numpy()
        preds_TMK = all_preds.transpose(1, 0, 2)
        if self.coeff_level:
            return np.sum(W * preds_TMK, axis=1).astype(np.float32)
        return np.sum(W[:, :, None] * preds_TMK, axis=1).astype(np.float32)


# ============================================================
# Latent decoder
# ============================================================
class LatentDecoder:
    def __init__(self, cae: COAE, stats_cae: Any, device: str, batch_size: int = 4):
        self.cae        = cae
        self.stats_cae  = stats_cae
        self.device     = device
        self.batch_size = batch_size

    @torch.no_grad()
    def decode(self, latent_series: np.ndarray) -> np.ndarray:
        loader = DataLoader(
            torch.tensor(latent_series, dtype=torch.float32),
            batch_size=self.batch_size,
            shuffle=False,
        )
        dens = [
            np.power(
                10,
                utils_cae.denormalize(
                    self.cae.decoder(batch.to(self.device)).detach().cpu().numpy(),
                    self.stats_cae,
                ),
            )
            for batch in loader
        ]
        return np.concatenate(dens, axis=0).squeeze(1)


# ============================================================
# Unscented Transform
# ============================================================
def _unscented_transform_density(
    base_latents_norm: np.ndarray,   # (M, H-1, K)
    mu_latents_full:   np.ndarray,   # (H, K) physical
    init_lat_phys:     np.ndarray,   # (K,)
    normalizer:        FeatureNormalizer,
    decoder:           LatentDecoder,
) -> tuple[np.ndarray, np.ndarray]:
    M, _, K = base_latents_norm.shape
    H       = mu_latents_full.shape[0]

    base_phys    = normalizer.denorm_latents(
        base_latents_norm.reshape(-1, K)
    ).reshape(M, H - 1, K).astype(np.float32)
    init_rep     = np.repeat(init_lat_phys[None, None, :], M, axis=0)
    lat_full_all = np.concatenate([init_rep, base_phys], axis=1)   # (M, H, K)

    alpha, beta, kappa = 1.0, 2.0, 0.0
    lam = alpha ** 2 * (K + kappa) - K
    c   = K + lam
    Wm    = np.full(2 * K + 1, 1.0 / (2.0 * c), dtype=np.float32)
    Wc    = np.full(2 * K + 1, 1.0 / (2.0 * c), dtype=np.float32)
    Wm[0] = lam / c
    Wc[0] = lam / c + (1.0 - alpha ** 2 + beta)

    sigma_lat          = np.zeros((H, 2 * K + 1, K), dtype=np.float32)
    sigma_lat[:, 0, :] = mu_latents_full

    for t in range(H):
        Xt = lat_full_all[:, t, :].astype(np.float32)
        mt = mu_latents_full[t].astype(np.float32)
        d  = Xt - mt[None, :]
        Pt = (d.T @ d) / float(max(M - 1, 1))
        Pt = 0.5 * (Pt + Pt.T) + 1e-6 * np.eye(K, dtype=np.float32)
        try:
            S = np.linalg.cholesky(c * Pt).astype(np.float32)
        except np.linalg.LinAlgError:
            Pt += 1e-3 * np.eye(K, dtype=np.float32)
            S   = np.linalg.cholesky(c * Pt).astype(np.float32)
        for i in range(K):
            sigma_lat[t, 1 + i,     :] = mt + S[:, i]
            sigma_lat[t, 1 + K + i, :] = mt - S[:, i]

    lat_flat  = sigma_lat.reshape(H * (2 * K + 1), K)
    dens_flat = decoder.decode(lat_flat)
    dens_sig  = dens_flat.reshape(H, 2 * K + 1, *dens_flat.shape[1:]).astype(np.float32)

    ut_mean = np.tensordot(Wm, dens_sig, axes=(0, 1)).astype(np.float32)
    diff    = dens_sig - ut_mean[:, None, ...]
    ut_var  = np.tensordot(Wc, diff * diff, axes=(0, 1)).astype(np.float32)

    return ut_mean, np.sqrt(np.maximum(ut_var, 0.0)).astype(np.float32)


# ============================================================
# ROPE_WAM  (IC-table mode, meta fusion only)
# ============================================================
class ROPE_WAM:
    """
    ROPE for WAM — IC-table mode, meta-model fusion only.

        rope = ROPE_WAM()
        out  = rope.run("2020-03-05 06:00:00", horizon=120)

    True density for evaluation:

        true_dens = rope.get_true_density("2020-03-05 06:00:00", horizon=120)
        # returns (H, 72, 36, 45) float32
    """

    def __init__(
        self,
        device:            str   = "cuda",
        decode_batch_size: int   = 4,
        use_xla:           bool  = False,

        # ── Stats / COAE ──────────────────────────────────────
        stats_ts_path:    str = os.path.join("data", "stats_ts_wam_f10avg.pt"),
        stats_cae_path:   str = os.path.join("data", "stats_cae_wam.pt"),
        coae_config_yaml: str = os.path.join("weights", "finetuned_coae_wam", "config.yaml"),
        coae_weights_pth: str = os.path.join("weights", "finetuned_coae_wam", "best_weights.pth"),

        # ── Ensemble model dirs ───────────────────────────────
        lstm_dir:        str = os.path.join("Models_v2", "StormTuned", "LSTM MODELS"),
        gru_dir:         str = os.path.join("Models_v2", "StormTuned", "GRU MODELS"),
        transformer_dir: str = os.path.join("Models_v2", "StormTuned", "TRANSFORMER MODELS"),

        # ── Meta model ────────────────────────────────────────
        meta_model_path: str = os.path.join("Meta Models", "meta_model_regular.keras"),

        # ── IC-table mode paths ───────────────────────────────
        driver_csv:   str = os.path.join("data", "SW_WAM.csv"),
        ic_table_csv: str = os.path.join("data", "IC_Table_wam.csv"),

     ):
        self.device = _safe_device(device)
        self.cfg    = WAMROPEConfig(decode_batch_size=decode_batch_size)

        tf.config.optimizer.set_jit(bool(use_xla))

        # ── Stats ─────────────────────────────────────────────
        self.stats_ts = torch.load(stats_ts_path, map_location="cpu")
        try:
            self.stats_cae = torch.load(stats_cae_path, weights_only=True, map_location="cpu")
        except TypeError:
            self.stats_cae = torch.load(stats_cae_path, map_location="cpu")

        self.normalizer = FeatureNormalizer(self.stats_ts, latent_dim=self.cfg.latent_dim)

        # ── IC + driver pipeline (IC-table mode) ──────────────
        df = pd.read_csv(driver_csv)
        df["datetime"]  = pd.to_datetime(df["datetime"])
        self._driver_df = df

        ic_table        = pd.read_csv(ic_table_csv)
        self._ic_interp = ICTableInterpolator(ic_table)

        self._window_builder = DriverWindowBuilder()
        self._seq_builder    = SequenceBuilder(
            normalizer=self.normalizer,
            ic_interp=self._ic_interp,
            latent_dim=self.cfg.latent_dim,
            seq_len=self.cfg.seq_len,
        )


        # ── COAE ──────────────────────────────────────────────
        coae_cfg = _load_yaml(coae_config_yaml)
        self.cae = COAE(config=coae_cfg.get("model"))
        try:
            sd = torch.load(coae_weights_pth, weights_only=True, map_location="cpu")
        except TypeError:
            sd = torch.load(coae_weights_pth, map_location="cpu")
        sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
        self.cae.load_state_dict(sd)
        self.cae.to(self.device).eval()

        # ── Ensemble models (base models feeding the meta fusion) ──
        self._all_models = (
            _load_keras_ensemble(lstm_dir, n_models=5)
            + _load_keras_ensemble(gru_dir, n_models=5)
            + _load_keras_ensemble(
                transformer_dir, n_models=5,
                custom_objects={"PositionalEncoding": PositionalEncoding},
            )
        )
        self._infer_fns = [_make_infer_fn(m) for m in self._all_models]

        # ── Meta model ──────────────────────────────────────────
        self._meta_infer_fn = _make_infer_fn(
            keras.models.load_model(meta_model_path, compile=False)
        )

        # ── Shared pipeline components ────────────────────────
        self._rollout = DynamicRollout(latent_dim=self.cfg.latent_dim)
        self._fuser   = MetaFusion(coeff_level=True)
        self._decoder = LatentDecoder(
            self.cae, self.stats_cae,
            device=self.device,
            batch_size=self.cfg.decode_batch_size,
        )

        # ── Public alias for downstream eval pipelines ────────
        self.driver_df = self._driver_df

        print(f"\nROPE_WAM ready  [ic=table  device={self.device}  decode_bs={decode_batch_size}]\n")

    # ----------------------------------------------------------
    def _prepare(
        self, start_datetime, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Returns (x_chunk_np, X_init_norm, start_idx)."""
        internal_df = self._window_builder.build(
            self._driver_df, start_datetime,
            horizon=horizon, seq_len=self.cfg.seq_len,
        )
        start_idx   = self.cfg.seq_len - 1
        forecast_df = internal_df.iloc[start_idx: start_idx + horizon].reset_index(drop=True)
        X_init_norm = self._seq_builder.build_X_init_norm(internal_df.iloc[: self.cfg.seq_len])
        x_chunk_np  = self._seq_builder.build_x_chunk(X_init_norm, forecast_df, horizon)
        self._last_forecast_df = forecast_df
        return x_chunk_np, X_init_norm, start_idx

    # ----------------------------------------------------------
    def _build_window_df(self) -> pd.DataFrame:
        cols = [c for c in ["datetime", "f10", "f10_41day_avg", "ap"]
                if c in self._last_forecast_df.columns]
        return self._last_forecast_df[cols].copy()



    # ----------------------------------------------------------
    def run(
        self,
        start_datetime,
        horizon:     int  = 120,
        uncertainty: bool = True,
    ) -> Dict[str, Any]:
        """
        Run WAM-ROPE forecast (IC-table mode, meta fusion).

        Parameters
        ----------
        start_datetime : str | datetime | pd.Timestamp
        horizon        : forecast steps in hours
        uncertainty    : compute density_std via Unscented Transform

        Returns
        -------
        dict:
            window_df    : pd.DataFrame  (datetime, f10, f10_41day_avg, ap)
            mean_density : np.ndarray    (H, 72, 36, 45)
            density_std  : np.ndarray    (H, 72, 36, 45)  if uncertainty=True
        """
        H = int(horizon)
        K = self.cfg.latent_dim

        # 1) Build x_chunk + IC
        x_chunk_np, X_init_norm, start_idx = self._prepare(start_datetime, H)

        # 2) Run 15 base models  →  (M, H-1, K)
        base_latents_norm = np.stack(
            [self._rollout.run(fn, x_chunk_np, H) for fn in self._infer_fns],
            axis=0,
        ).astype(np.float32)

        # 3) Fuse via meta model  →  (H-1, K) normalized
        fused_norm = self._fuser.fuse(
            self._meta_infer_fn, x_chunk_np[: H - 1], base_latents_norm,
        )

        # 4) Prepend true t=0  →  (H, K) physical
        init_lat_phys = self.normalizer.denorm_latents(X_init_norm[-1, :K])
        fused_full    = np.vstack([
            init_lat_phys[None, :],
            self.normalizer.denorm_latents(fused_norm),
        ])

        # 5) Decode mean
        density = self._decoder.decode(fused_full)

        out: Dict[str, Any] = {
            "window_df":    self._build_window_df(),
            "mean_density": density,
        }

        # 6) Uncertainty via Unscented Transform
        if uncertainty:
            ut_mean, ut_std = _unscented_transform_density(
                base_latents_norm=base_latents_norm,
                mu_latents_full=fused_full,
                init_lat_phys=init_lat_phys,
                normalizer=self.normalizer,
                decoder=self._decoder,
            )
            out["mean_density"] = ut_mean
            out["density_std"]  = ut_std

        return out