## 二分类FER

* 数据集相关
  * 构造训练集
    * ExpW数据、fer2013数据、multiPIE数据、补充训练数据、在celebrity数据微调
  * 构造评测集
    * 在celebrity数据集上构建评测集
* 模型选用：
  * 基础结构：~~res_50第一层+pooling+修改的res_block~~   alex-net去掉一层最后卷积，使用全局Pooling接分类层
  * 结构修改：~~在res_block基础上将最后的卷积核个数由256调整至32~~    调整对应的卷积核数量（实验证明基于Alex的修改分类性能显著优于res-block的模型）

* 训练测试code相关
  * 训练[执行](./code/train_2classify.sh)、[solver](./code/size32_2classify_solver.prototxt)、[训练网络proto](./code/size32_2classify_train_val_v7_alex.prototxt)
  * [python多进程FER二分类](./code/classify.py)，并基于测试得分进行[数据分析](./code/data_analysis.py)及数据进一步筛选。
  * C++接口 Linux版本FER二分类（[分类代码](./code/SsFer_thre.cpp)、[编译](./code/complie.sh)、[运行脚本](./code/run.sh)、[测试模型](./code/v7_param.caffemodel)对应地[数组格式](./code/v7_param.h)、[测试网络proto](./code/v7_net.prototxt)对应地[数组格式](./code/v7_net.h)）