"""Keep the DAQ field calibration of imaging and EDS in step with the SEM magnification.

The SEM scales its external-scan input with the magnification, so the field
the DAQ covers at full-scale volts is ``K / magnification`` with one constant
``K`` (µm, stored in :class:`~easyebl.microscope.settings.MicroscopeSettings`).

* If ``K`` is unknown, it is taken from the current field calibration the first
  time the magnification is known (the calibration is assumed to be valid at
  that magnification) – recalibrate the field once to make it exact.
* Whenever the magnification changes (on the SEM panel, from the SEM window's
  wheel, from the microscope window), every target gets
  ``field_size_um = K / magnification``.
* :meth:`FieldFollower.adopt` re-derives ``K`` after the user calibrated the
  field by hand (SEM window → Advanced → Calibration).

Targets are objects with ``calibration`` and ``set_calibration(cal)`` – the
imaging engine and the EDS controller.  With the simulator backend the
simulated SEM's field is scaled too, so the simulated image zooms like the
real one.
"""
from __future__ import annotations

from dataclasses import replace
import threading

from .control import MicroscopeControl

__all__ = ["FieldFollower"]


class FieldFollower:
    def __init__(self, microscope: MicroscopeControl, targets, settings=None, sim_backend=None, notify=None):
        self.mic = microscope
        self.targets = [t for t in targets if t is not None]
        self.settings = settings
        self.sim_backend = sim_backend
        link = getattr(microscope, "link", None)
        self.notify = notify or (lambda msg: link.events.emit("info", msg) if link is not None else None)
        self._lock = threading.Lock()
        self._mag = None
        self.enabled = bool(getattr(settings, "follow_magnification", True))
        microscope.add_listener(self._changed)
        m = microscope.magnification()
        if m:
            self._apply(m)

    @property
    def k_um(self) -> float | None:
        return getattr(self.settings, "field_mag_product_um", None) if self.settings is not None else self._k

    _k = None

    def _set_k(self, k: float) -> None:
        if self.settings is not None:
            self.settings.field_mag_product_um = float(k)
            try:
                self.settings.save()
            except OSError:
                pass
        else:
            self._k = float(k)

    def field_for(self, mag: float) -> float | None:
        k = self.k_um
        return None if not k or not mag else k / float(mag)

    def _changed(self, changes: dict) -> None:
        if "mag" in changes and changes["mag"]:
            self._apply(changes["mag"])

    def _apply(self, mag: float) -> None:
        with self._lock:
            self._mag = float(mag)
            if not self.targets:
                return
            if not self.k_um:
                f0 = self.targets[0].calibration.field_size_um
                self._set_k(f0 * self._mag)
                self.notify(f"field calibration tied to the magnification: {f0:g} µm at ×{self._mag:g} "
                            f"(K = {self.k_um:.4g} µm); recalibrate the field once to make it exact")
                self._sim(f0)
                return
            if not self.enabled:
                return
            field = self.k_um / self._mag
            for t in self.targets:
                cal = t.calibration
                if abs(cal.field_size_um - field) > 1e-9 * field:
                    self._set(t, replace(cal, field_size_um=field))
            self._sim(field)
        self.notify(f"×{mag:g}: field {field:g} µm")

    @staticmethod
    def _set(target, cal) -> None:
        """Imaging zooms with the magnification (the scan keeps its share of the field)."""
        try:
            target.set_calibration(cal, scale_scan=True)
        except TypeError:  # targets without a scan (EDS controller)
            target.set_calibration(cal)

    def _sim(self, field: float) -> None:
        """Simulator: the simulated SEM scales its scan with the magnification too."""
        be = self.sim_backend
        if be is not None and hasattr(be, "params"):
            be.params.field_size_um = float(field)

    def adopt(self) -> None:
        """The field was calibrated by hand at the current magnification: derive K from it."""
        with self._lock:
            if self._mag and self.targets:
                f = self.targets[0].calibration.field_size_um
                expected = self.field_for(self._mag)
                if expected is not None and abs(f - expected) <= 1e-6 * expected:
                    return  # unchanged
                self._set_k(f * self._mag)
                for t in self.targets[1:]:
                    if abs(t.calibration.field_size_um - f) > 1e-9 * f:
                        t.set_calibration(replace(t.calibration, field_size_um=f))
                self._sim(f)
                self.notify(f"field calibration: {f:g} µm at ×{self._mag:g} (K = {self.k_um:.4g} µm)")

    def close(self) -> None:
        self.mic.remove_listener(self._changed)
