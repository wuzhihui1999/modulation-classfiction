# modulation-classfiction
A new method named RLDNN and better accuracy

基于公用数据集Radioml2016.10a在CLDNN网络结构基础上增加了跨层连接，在LSTM层中增加了一层卷积，并在数据训练前进行星座相位旋转的数据增强，结果表明分类精度在0db以上几乎能到93%，比大多数论文中对同一数据集的效果更好一些。

实验环境是win10，python3.8，keras2.8，tensorflow2.8，显卡是RTX3070(laptop)训练数据集是RadioML2016.10a公开数据集，一共有22万条IQ时域信号数据共11种调制类型，其中8种数字调制，3种模拟调制，训练集和测试集比例为1：1，各占50%，损失函数采用交叉熵，batchsize设定为1024，优化方法选用Adam，学习率初始为0.001，训练集从随机种子为2016的随机数中抽取，中间的dropout比例选取0.5，对训练集训练100轮后在所有测试集上测试，统计每种信噪比下的综合识别精度，并画出混淆矩阵。
