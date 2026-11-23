"""The global field calibration: one constant for every column magnification.

The SEM scales its external-scan input with the magnification, so the field the
DAQ sweeps with the full deflection range (±full-scale volts) is

    field (µm) = K / magnification

with one constant ``K``: the field at a hypothetical ×1 (equivalently
``K / (2 × full scale)`` µm per volt at ×1).  :class:`FieldScale` owns ``K``
and keeps the field calibration of its targets (the imaging engine, the EDS
controller) at ``K / magnification``:

* the magnification comes from the microscope; when it reports none (not
  connected) the *manual* magnification is used – entered by the user;
* ``K`` is calibrated by a line of known length measured on the image
  (:meth:`calibrate`) or set directly (:meth:`set_k`);
* with settings (:class:`~easyebl.microscope.settings.MicroscopeSettings`) ``K``
  and the manual magnification are stored on disk; without, they live in memory
  (the simulators – they must not overwrite the instrument's calibration).

Targets are objects with ``calibration`` and ``set_calibration(cal)``.  With the
simulator backend the simulated SEM's field is scaled too, so the simulated
image zooms like the real one.
"""
from __future__ import annotations

from dataclasses import replace
import threading

from .control import MicroscopeControl, StubMicroscope

__all__ = ["FieldScale", "DEFAULT_FIELD_AT_1X_UM", "DEFAULT_MANUAL_MAG"]

DEFAULT_FIELD_AT_1X_UM = 100_000.0  # 100 µm at ×1000 – until calibrated
DEFAULT_MANUAL_MAG = 1000.0


class FieldScale:
    def __init__(self, microscope: MicroscopeControl | None, targets, settings=None, k_um: float | None = None,
                 sim_backend=None, notify=None):
        self.mic = microscope if microscope is not None else StubMicroscope()
        self.targets = [t for t in targets if t is not None]
        self.settings = settings
        self.sim_backend = sim_backend
        link = getattr(self.mic, "link", None)
        self.notify = notify or (lambda msg: link.events.emit("info", msg) if link is not None else None)
        self._lock = threading.RLock()
        if settings is not None:
            self._k = float(settings.field_at_1x_um)
            self._manual = float(settings.manual_magnification)
        else:
            self._manual = DEFAULT_MANUAL_MAG
            if k_um is None and self.targets:  # in memory: keep the targets' present field
                k_um = self.targets[0].calibration.field_size_um * self.magnification
            self._k = float(k_um or DEFAULT_FIELD_AT_1X_UM)
        self.mic.add_listener(self._changed)
        self.apply()

    # ------------------------------------------------------------------ readout
    @property
    def k_um(self) -> float:
        """Field (µm, full deflection range) at ×1."""
        return self._k

    @property
    def from_microscope(self) -> bool:
        """The magnification is reported by the microscope (else the manual one is used)."""
        return bool(self.mic.magnification())

    @property
    def manual_magnification(self) -> float:
        return self._manual

    @property
    def magnification(self) -> float:
        return float(self.mic.magnification() or self._manual)

    @property
    def field_um(self) -> float:
        """Field at the current magnification."""
        return self._k / self.magnification

    @property
    def v_full_scale(self) -> float:
        return self.targets[0].calibration.v_full_scale if self.targets else 10.0

    def um_per_volt(self, magnification: float | None = None) -> float:
        """µm of beam travel per deflection volt at *magnification* (default: the current one)."""
        return self._k / float(magnification or self.magnification) / (2.0 * self.v_full_scale)

    # ------------------------------------------------------------------ changes
    def set_k(self, k_um: float) -> None:
        if not k_um or k_um <= 0:
            raise ValueError("the field at ×1 must be positive")
        with self._lock:
            self._k = float(k_um)
            self._save()
            self.apply()
        self.notify(f"field calibration: {self._k:.6g} µm at ×1 → {self.field_um:.4g} µm at ×{self.magnification:g}")

    def set_manual_magnification(self, mag: float) -> None:
        """The magnification used while the microscope does not report one."""
        if not mag or mag <= 0:
            raise ValueError("the magnification must be positive")
        with self._lock:
            self._manual = float(mag)
            self._save()
            self.apply()

    def calibrate(self, measured_um: float, true_um: float) -> float:
        """A line measured *measured_um* long (with the present calibration) is really *true_um*:
        scale ``K`` by true / measured.  Returns that ratio."""
        if not measured_um or measured_um <= 0 or not true_um or true_um <= 0:
            raise ValueError("both lengths must be positive")
        r = float(true_um) / float(measured_um)
        self.set_k(self._k * r)
        return r

    def _save(self) -> None:
        s = self.settings
        if s is None:
            return
        s.field_at_1x_um, s.manual_magnification = self._k, self._manual
        try:
            s.save()
        except OSError:
            pass

    def _changed(self, changes: dict) -> None:
        if changes.get("mag"):
            self.apply()

    def apply(self) -> None:
        """Every target to ``K / magnification``."""
        with self._lock:
            field = self.field_um
            for t in self.targets:
                cal = t.calibration
                if abs(cal.field_size_um - field) > 1e-9 * field:
                    self._set(t, replace(cal, field_size_um=field))
            be = self.sim_backend  # the simulated SEM scales its scan with the magnification too
            if be is not None and hasattr(be, "params"):
                be.params.field_size_um = float(field)

    @staticmethod
    def _set(target, cal) -> None:
        """Imaging zooms with the magnification (the scan keeps its share of the field)."""
        try:
            target.set_calibration(cal, scale_scan=True)
        except TypeError:  # targets without a scan (EDS controller)
            target.set_calibration(cal)

    def close(self) -> None:
        self.mic.remove_listener(self._changed)
