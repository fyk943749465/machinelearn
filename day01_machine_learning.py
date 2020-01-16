from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

import jieba
import pandas as pd


def datasets_demo():
    iris = load_iris()
    print("鸢尾花数据集:\n", iris)
    # 返回值是一个继承自字典的Bunch
    print("鸢尾花特征值:\n", iris["data"])
    print("鸢尾花目标值:\n", iris.target)
    print("鸢尾花特征名字:\n", iris.feature_names)
    print("鸢尾花目标的名字：\n", iris.target_names)
    print("鸢尾花的描述:\n", iris.DESCR)

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值:\n", x_train, x_train.shape)
    return None


# 特征抽取，处理成one-hot编码
# 数据集当中，类类别数据比较多的时候，需要使用特征抽取
def dict_demo():
    """
    字典特征抽取
    :return:
    """
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30},
            {'city': '长治', 'temperature': 30}]
    # 1 实例化一个转化器类 sparse值默认是True，节省内存
    transfer = DictVectorizer()
    # 2 调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("特征名字:\n", transfer.get_feature_names())
    return None


def count_demo():
    """
    文本特征抽取
    :return:
    """
    data = ["lift is short, i like like python", "life is too long, i dislike python"]
    # 实例化一个转换器类
    transfer = CountVectorizer()
    # 调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字：", transfer.get_feature_names())
    return None


# 如果是中文，需要用专门的分词库来分词
def count_chinese_demo():
    """
    文本特征抽取
    :return:
    """
    data = ["我爱北京天安门", "天安门前太阳升"]
    # 实例化一个转换器类
    transfer = CountVectorizer()
    # 调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字：", transfer.get_feature_names())
    return None


def cut_word(text):
    """
    进行中文分词：“我爱北京天安门”-->"我 爱 北京 天安门"
    :param text:
    :return:
    """
    return " ".join(list(jieba.cut(text)))


def count_chinese_demo2():
    """
    中文文本特征抽取/自动分词
    :return:
    """

    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝大多数是死在明天晚上。所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去",
            "如果只用一种方式了解某种事物，你就不会真正了解它。了解事物真正含义取决于如果将其与我们所了解的事物相联系。"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))

    transfer = CountVectorizer(stop_words=["一种", "所以", "不要"])

    data_final = transfer.fit_transform(data_new)

    print("data_final:\n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names())

    return None


def tfidf_demo():
    """
    用TF-IDF的方法进行文本特征抽取
    :return:
    """

    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝大多数是死在明天晚上。所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去",
            "如果只用一种方式了解某种事物，你就不会真正了解它。了解事物真正含义取决于如果将其与我们所了解的事物相联系。"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))

    transfer = TfidfVectorizer(stop_words=["一种", "所以", "不要"])

    data_final = transfer.fit_transform(data_new)

    print("data_final:\n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names())
    return None


def minmax_demo():
    """
    数据归一化
    :return:
    """
    # 1.获取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]
    # 2.实例化一个转换器类
    transfer = MinMaxScaler()

    # 3.调用fit_transform转化
    data_new = transfer.fit_transform(data)
    return None


def stand_demo():
    """
    标准化
    :return:
    """
    # 1.获取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]
    # 2.实例化一个转换器类
    transfer = StandardScaler()

    # 3.调用fit_transform
    data_new = transfer.fit_transform(data)

    return None


def variance_demo():
    """
    过滤低方差特征
    :return:
    """
    # 1. 获取数据
    data = pd.read_csv("factor_return.csv")
    # 选择所需要的数据，所有行，从第二列开始到倒数第二列
    data = data.iloc[:, 1:-2]
    # 2. 实例化一个转化器类 threshold用来过滤低方差特征
    transfer = VarianceThreshold(threshold=0)
    # 3. 调用fit_transfer
    data_new = transfer.fit_transform(data)
    print("返回的数据\n", data_new)

    # 计算某两个变量的相关系数
    r = pearsonr(data["pe_ratio"], data["pb_ratio"])
    print("相关系数\n", r)
    return None


if __name__ == "__main__":
    # sklearn数据集使用
    # datasets_demo()
    # 字典特征抽取
    # dict_demo()
    # 文本特征抽取
    # count_chinese_demo()
    # count_chinese_demo2()
    tfidf_demo()
