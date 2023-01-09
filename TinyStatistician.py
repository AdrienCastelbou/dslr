from math import sqrt
import numpy as np
from typing import Union

class TinyStatistician:

    @staticmethod
    def count(x):
        try:
            l = len(x)
            res = 0.0
            for elem in x:
                if not isinstance(elem, (float, int)):
                    return None
                res += elem
            return res
        except:
            return None

    @staticmethod
    def mean(x: Union[list, np.array]) -> float:
        try:
            l = len(x)
            res = 0.0
            for elem in x:
                if not np.isreal(elem):
                    print(elem, type(elem))
                    return None
                res += elem
            return res / l
        except:
            return None

    @staticmethod
    def median(x: Union[list, np.array]) -> float:
        try:
            sorted_x = sorted(x)
            l = len(x)
            if l == 0 or not all(np.isreal(elem) for elem in x):
                return None
            if l % 2:
                return sorted_x[int(len(x) / 2)]
            else:
                return (sorted_x[int(len(x) / 2) - 1] + sorted_x[int(len(x) / 2)]) / 2
        except:
            return None

    @staticmethod
    def quartile(x: Union[list, np.array]):
        try:
            l = len(x)
            if l == 0 or not all(np.isreal(elem) for elem in x):
                return None
            sorted_x = sorted(x)
            print(x)
            return [float(sorted_x[int(l / 4)]), float(sorted_x[int(l * 3 / 4)])]
        except:
            return None

    @staticmethod
    def percentile(x: Union[list, np.array], p: Union[int, float]) -> float:
        try:
            l = len(x)
            sorted_x = sorted(x)
            r = p / 100 * (l- 1)
            rr = r - np.floor(r)
            return sorted_x[int(np.floor(r))] + rr * (sorted_x[int(np.ceil(r))] - sorted_x[int(np.floor(r))])
        except:
            return None

    @staticmethod
    def var(x: Union[list, np.array]) -> float:
        try:
            l = len(x)
            if l == 0 or not all(np.isreal(elem) for elem in x):
                return None
            mean = TinyStatistician.mean(x)
            res = 0
            for elem in x:
                res += (elem - mean) * (elem - mean)
            return res / (l  - 1)
        except:
            return None


    @staticmethod
    def std(x: Union[list, np.array]) -> float:
        try:
            l = len(x)
            if l == 0 or not all(np.isreal(elem) for elem in x):
                return None
            return sqrt(TinyStatistician.var(x))
        except:
            return None