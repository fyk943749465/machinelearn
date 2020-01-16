import numpy as np
import pandas as pd
from pandas import Series, DataFrame


def pivotTab():
    """
    透视表
    :return:
    """
    print(df)
    print(df.pivot_table(index=['产地', '类别']))  # 产地和类别为列
    print(df.pivot_table(columns=['产地', '类别']))  # 产地和类别为行
    print(df.pivot_table('价格', index='产地', columns='类别', aggfunc='max', margins=True, fill_value=0))


def crossTab():
    """
    交叉表
    :return:
    """
    print(pd.crosstab(df['类别'], df['产地'], margins=True))


if __name__ == "__main__":

    df = DataFrame({'类别': ['水果', '水果', '水果', '蔬菜', '蔬菜', '肉类', '肉类'],
                    '产地': ['美国', '中国', '中国', '中国', '新西兰', '新西兰', '美国'],
                    '水果': ['苹果', '梨', '草莓', '番茄', '黄瓜', '羊肉', '牛肉'],
                    '数量': [5, 5, 9, 3, 2, 10, 8],
                    '价格': [5, 5, 10, 3, 3, 13, 20]})
    crossTab()
