"""``python -m easyebl.eds.gui.suite``: SEM imaging + EDS in one program (same as ``easyebl-suite``)."""
from .app import suite_main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(suite_main())
