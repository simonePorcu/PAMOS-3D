"""
pamos3d.adapters

Format adapters turn format-specific source material into PRCM objects
without the rest of the library needing to know about that format.

Only the point-cloud adapter is implemented today
(pamos3d.adapters.pointcloud, for V-PCC / G-PCC Octree / G-PCC Trisoup).
A Gaussian Splat or mesh adapter would live alongside it as a new module
with the same kind of functions, feeding a D2AN instance retrained for
that format.

Author: Simone Porcu
"""
