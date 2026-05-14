二分类BP神经网络示例代码：
1. Python文件夹中，为Python仿真代码，展示了训练模型并进行推理的过程。所使用的数据由Python代码自动生成，每次运行Python代码都会生成新的随机数据，数据和标签存储在feature.csv和label.csv文件中（若已有feature.csv和label.csv文件，会直接覆盖）。Python文件运行时会打印出生成的数据、训练进度、模型结构和模型各层参数，模型相关信息在C代码实现时通常会使用到。
2. C文件夹中，为C代码，展示了模型推理的过程。由于Python每次会重新训练模型，若要比对Python代码和C代码的一致性，可将Python运行时打印出的模型系数 input_layer.weight、input_layer.bias、hidden_layer.weight、hidden_layer.bias 相应赋值到C代码的数组中，即alg_bp_networks.c的input_layer_weight、input_layer_bias、hidden_layer_weight、hidden_layer_bias数组中。同学们可自行尝试替换C代码参数，并比对与Python代码结果，来熟悉Python模型转C代码的过程。
3. 若自行配置的环境运行异常，可参考如下版本进行环境配置：
    - C代码编译：gcc (GCC) 12.2.0；
    - Python代码运行：Python 3.10.5；第三方库（matplotlib==3.10.9，numpy==1.26.4，pandas==2.3.3，torch==2.3.1+cpu，torchaudio==2.3.1+cpu，torchvision==0.18.1+cpu）。
