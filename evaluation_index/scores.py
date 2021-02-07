# some function
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


def get_score(y_ture, y_pred, average="macro"):
    """[summary]

    Args:
        y_ture ([type]): [description]
        y_pred ([type]): [description]
        average (str, optional): [description]. Defaults to "macro".
            macro
            binary: 二分类

    Returns:
        [type]: [description]
        p： 精确度
        f1：

    """
    y_ture = np.array(y_ture)
    y_pred = np.array(y_pred)
    f1 = f1_score(y_ture, y_pred, average=average) * 100
    p = precision_score(y_ture, y_pred, average=average) * 100
    r = recall_score(y_ture, y_pred, average=average) * 100

    return str((reformat(p, 2), reformat(r, 2), reformat(f1, 2))), reformat(f1, 2)


def reformat(num, n):
    """[对于num， 取小数点后n位]

    Args:
        num ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    return float(format(num, '0.' + str(n) + 'f'))

if __name__ == "__main__":
    n = 4
    num = 0.00010001
    print(format(num, '0.' + str(n) + 'f'))