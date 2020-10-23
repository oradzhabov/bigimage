import os
import numpy as np
from .. import get_submodules_from_kwargs


# About how to find measure Precision-Recall curve (and best Threshold, which corresponds to best point on curve)
# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

def get_pr_recall_metrics(**kwargs):
    _backend, _layers, _models, _keras_utils, _optimizers, _legacy, _callbacks = get_submodules_from_kwargs(kwargs)

    # Setup framework for segmentation package
    sm_framework = 'keras' if _legacy is not None else 'tf.keras'
    os.environ["SM_FRAMEWORK"] = sm_framework
    import segmentation_models as sm

    thr_step = 0.002
    metrics = list()
    for threshold in np.arange(0.48, 0.52+thr_step, thr_step):
        sub_name = '{:.3f}'.format(threshold)
        metrics = metrics + [sm.metrics.Precision(threshold=threshold, name='pr_{}'.format(sub_name)),
                             sm.metrics.Recall(threshold=threshold, name='re_{}'.format(sub_name))]
    return metrics


import numpy as np
import matplotlib.pyplot as plt


fname = '3.csv'
data = np.genfromtxt(fname, delimiter=",", names=["th", "pr", "re"])

r = dict()
for i, th in enumerate(data['th']):
    r['pr_{:.3f}'.format(th)] = data['pr'][i]
    r['re_{:.3f}'.format(th)] = data['re'][i]
    r['th_{:.3f}'.format(th)] = th

print(r)
r_keys = list(r.keys())
r_values = list(r.values())
print(r_keys)
print(r_values)

# ========================================================================================
rr = dict()
for i in range(len(r_keys)):
    metric_name = r_keys[i]
    value = r_values[i]
    if metric_name[:3] == 'pr_' or metric_name[:3] == 're_':
        th = int(float(metric_name[3:]) * 1000)
        key = metric_name[:2]
        if th not in rr:
            rr[th] = dict()
        rr[th][key] = value
print('rr', rr)

th_list = list(rr.keys())
th_list.sort()
data = dict({'th': list(), 'pr': list(), 're': list()})
for k in th_list:
    data['th'].append(str(k))
    data['pr'].append(rr[k]['pr'])
    data['re'].append(rr[k]['re'])


plt.plot(data['pr'], data['re'], label='Threshold value')
for i in range(len(data['th'])):
    a = np.array((data['pr'][i], data['re'][i])) - np.array((1, 1))
    dist = np.linalg.norm(a)
    text = 'th{}_d{:.5f}'.format(data['th'][i], dist)
    plt.annotate(text, xy=(data['pr'][i], data['re'][i]))
plt.xlabel('Precision')
plt.ylabel('Recall')

plt.plot([0, 1], [0, 1], label='0-1')
plt.legend()
plt.show()
