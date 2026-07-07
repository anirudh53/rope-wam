"""
density_interpolator_wam.py  —  Spatial/temporal interpolator for WAM-ROPE output
-----------------------------------------------------------------------------------
Grid axes:
    LST : np.linspace(0, 24, 72)     [hours]   — axis already covers full cycle
    LAT : np.linspace(-88, 88, 36)   [degrees]
    ALT : np.linspace(100, 1000, 45) [km]

Altitude handling (simple):
    - alt within [100, 1000] km : trilinear interpolation
    - alt outside grid          : returns 0.0  (no extrapolation)

No cyclic LST extension needed — axis already runs 0 → 24.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Union, Optional
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


class TimeOutOfRangeError(ValueError):
    pass


@dataclass(frozen=True)
class GridAxes:
    lst_axis: np.ndarray
    lat_axis: np.ndarray
    alt_axis: np.ndarray


def default_axes() -> GridAxes:
    return GridAxes(
        lst_axis=np.linspace(0, 24, 72),
        lat_axis=np.linspace(-88, 88, 36),
        alt_axis=np.linspace(100, 1000, 45),
    )


class DensityInterpolator:
    """
    Query WAM-ROPE output at a single timestamp and spatial coordinate.

    Density source priority:
      1) res["mean_density"]   (WAM-ROPE primary key)
      2) res["meta_density"]   (fallback for cross-compatibility)

    Uncertainty:
      If res["density_std"] exists, query() also returns "sigma".

    time_mode:
      - "hold_next_hour" : use nearest model hour, no time interpolation
      - "interp_time"    : linear interpolation between bracketing snapshots

    Out-of-bounds → returns 0.0 with a warning. No extrapolation.
    """

    def __init__(
        self,
        res: Dict[str, Any],
        axes: Optional[GridAxes] = None,
    ):
        if "window_df" not in res or "datetime" not in res["window_df"].columns:
            raise KeyError("res must contain 'window_df' with a 'datetime' column")

        # ── Density ───────────────────────────────────────────
        if "mean_density" in res:
            self.dens = np.asarray(res["mean_density"])
        elif "meta_density" in res:
            self.dens = np.asarray(res["meta_density"])
        else:
            raise KeyError("res must contain 'mean_density' or 'meta_density'")

        self.times = pd.to_datetime(res["window_df"]["datetime"]).reset_index(drop=True)

        if self.dens.ndim != 4 or self.dens.shape[1:] != (72, 36, 45):
            raise ValueError(
                f"density must have shape (T, 72, 36, 45). Got {self.dens.shape}"
            )
        if len(self.times) != self.dens.shape[0]:
            raise ValueError(
                f"Time mismatch: len(window_df)={len(self.times)}, "
                f"density T={self.dens.shape[0]}"
            )

        # ── Uncertainty ───────────────────────────────────────
        self.sigma = None
        if "density_std" in res:
            self.sigma = np.asarray(res["density_std"])
            if self.sigma.shape != self.dens.shape:
                raise ValueError(
                    f"density_std shape {self.sigma.shape} != density shape {self.dens.shape}"
                )

        self.axes = axes if axes is not None else default_axes()

        self._lst_min = float(self.axes.lst_axis.min())
        self._lst_max = float(self.axes.lst_axis.max())
        self._lat_min = float(self.axes.lat_axis.min())
        self._lat_max = float(self.axes.lat_axis.max())
        self._alt_min = float(self.axes.alt_axis.min())
        self._alt_max = float(self.axes.alt_axis.max())

        self._t_min = self.times.iloc[0]
        self._t_max = self.times.iloc[-1]

    # ── bounds ────────────────────────────────────────────────
    def bounds(self) -> Dict[str, Any]:
        return {
            "lst":  (self._lst_min, self._lst_max),
            "lat":  (self._lat_min, self._lat_max),
            "alt_km": (self._alt_min, self._alt_max),
            "time": (self._t_min, self._t_max),
        }

    # ── helpers ───────────────────────────────────────────────
    def _warn(self, msg: str) -> None:
        warnings.warn(msg, category=RuntimeWarning, stacklevel=3)

    def _wrap_lst(self, lst: float) -> float:
        """Wrap LST into [0, 24). 24.0 → 0.0, -0.5 → 23.5, etc."""
        wrapped = float(lst % 24.0)
        if wrapped == 24.0:
            wrapped = 0.0
        return wrapped

    def _spatial_is_in_bounds(self, lst: float, lat: float, alt_km: float) -> bool:
        lst_w = self._wrap_lst(lst)
        if not (self._lst_min <= lst_w <= self._lst_max):
            self._warn(f"LST {lst_w} out of [{self._lst_min}, {self._lst_max}]. Returning 0.")
            return False
        if not (self._lat_min <= lat <= self._lat_max):
            self._warn(f"lat {lat} out of [{self._lat_min}, {self._lat_max}]. Returning 0.")
            return False
        if not (self._alt_min <= alt_km <= self._alt_max):
            self._warn(f"alt_km {alt_km} out of [{self._alt_min}, {self._alt_max}]. Returning 0.")
            return False
        return True

    def _make_interpolator(self, field_t: np.ndarray) -> RegularGridInterpolator:
        return RegularGridInterpolator(
            (self.axes.lst_axis, self.axes.lat_axis, self.axes.alt_axis),
            field_t,
            bounds_error=False,
            fill_value=0.0,
        )

    def _spatial_value(self, field_t: np.ndarray, lst: float, lat: float, alt_km: float) -> float:
        lst_w = self._wrap_lst(lst)
        point = np.array([[lst_w, lat, alt_km]], dtype=np.float64)
        f = self._make_interpolator(field_t)
        return float(f(point)[0])

    def _bracket_indices(self, when: pd.Timestamp) -> tuple:
        i1 = int(np.searchsorted(self.times.values, np.datetime64(when)))
        i0 = i1 - 1
        return i0, i1

    # ── main API ──────────────────────────────────────────────
    def query(
        self,
        when: Union[str, pd.Timestamp],
        lst: float,
        lat: float,
        alt_km: float,
        time_mode: str = "hold_next_hour",
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        when      : datetime string or Timestamp
        lst       : Local Solar Time [hours]  0–24
        lat       : Geographic latitude [degrees]  -88 to 88
        alt_km    : Altitude [km]  100–1000
        time_mode : "hold_next_hour" | "interp_time"
        """
        when = pd.to_datetime(when)

        if time_mode not in ("hold_next_hour", "interp_time"):
            raise ValueError("time_mode must be 'hold_next_hour' or 'interp_time'")

        if when < self._t_min or when > self._t_max:
            raise TimeOutOfRangeError(
                f"Requested time {when} outside [{self._t_min}, {self._t_max}]"
            )

        lst_f      = float(lst)
        lat_f      = float(lat)
        alt_f      = float(alt_km)
        lst_w      = self._wrap_lst(lst_f)
        spatial_ok = self._spatial_is_in_bounds(lst_f, lat_f, alt_f)
        i0, i1    = self._bracket_indices(when)

        # ── hold_next_hour ────────────────────────────────────
        if time_mode == "hold_next_hour":
            use_i = i0 if when == self.times.iloc[i0] else i1

            dens_val = self._spatial_value(self.dens[use_i], lst_f, lat_f, alt_f) \
                       if spatial_ok else 0.0

            out = {
                "datetime_requested": when,
                "datetime_used":      self.times.iloc[use_i],
                "density":            dens_val,
                "t_index":            int(use_i),
                "time_mode":          "hold_next_hour",
                "spatial_oob":        not spatial_ok,
                "lst_wrapped":        lst_w,
            }
            if self.sigma is not None:
                out["sigma"] = self._spatial_value(self.sigma[use_i], lst_f, lat_f, alt_f) \
                               if spatial_ok else 0.0
            return out

        # ── interp_time ───────────────────────────────────────
        t0 = self.times.iloc[i0]
        t1 = self.times.iloc[i1]
        w  = float((when - t0) / (t1 - t0))

        if not spatial_ok:
            out = {
                "datetime":          when,
                "density":           0.0,
                "t_index_left":      int(i0),
                "t_index_right":     int(i1),
                "datetime_left":     t0,
                "datetime_right":    t1,
                "time_weight_right": w,
                "time_mode":         "interp_time",
                "spatial_oob":       True,
                "lst_wrapped":       lst_w,
            }
            if self.sigma is not None:
                out["sigma"] = 0.0
            return out

        d0 = self._spatial_value(self.dens[i0], lst_f, lat_f, alt_f)
        d1 = self._spatial_value(self.dens[i1], lst_f, lat_f, alt_f)

        out = {
            "datetime":          when,
            "density":           float((1.0 - w) * d0 + w * d1),
            "t_index_left":      int(i0),
            "t_index_right":     int(i1),
            "datetime_left":     t0,
            "datetime_right":    t1,
            "time_weight_right": w,
            "time_mode":         "interp_time",
            "spatial_oob":       False,
            "lst_wrapped":       lst_w,
        }
        if self.sigma is not None:
            s0 = self._spatial_value(self.sigma[i0], lst_f, lat_f, alt_f)
            s1 = self._spatial_value(self.sigma[i1], lst_f, lat_f, alt_f)
            out["sigma"] = float((1.0 - w) * s0 + w * s1)

        return out