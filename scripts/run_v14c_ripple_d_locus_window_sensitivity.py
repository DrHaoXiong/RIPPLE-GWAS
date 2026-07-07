#!/usr/bin/env python
"""Convenience entry point for RIPPLE-D V1.4c locus-window sensitivity.

This wrapper delegates to ``run_v14c_ripple_d_module_rescue.py`` so the module
statistics, nulls, and output schema remain identical across default and
window-sensitivity runs.
"""

from __future__ import annotations

import run_v14c_ripple_d_module_rescue


if __name__ == "__main__":
    run_v14c_ripple_d_module_rescue.main()
