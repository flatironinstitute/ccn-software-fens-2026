import pynaviz as viz
import pynapple as nap
import numpy as np
from PySide6.QtCore import QTimer

tsd = nap.Tsd(t=np.arange(1000), d=np.sin(np.arange(1000) * 0.1))
v = viz.TsdWidget(tsd)
QTimer.singleShot(5000, v.close)
v.show()
