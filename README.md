#  Transformer is all you need
#### 基于transformer的新冠肺炎确诊预测

众所周知，Transformer在时序预测中处于较为火的位置，本程序基于ConvTrans模型，通过对现有的新冠肺炎数据确诊病例预测，从而推断出相关城市未来疫情或套用其他城市模型的疫情走向。

```
注意：由于数据量过少，总计只有42天数据，数据维度只有天数和确诊病例，所以预测效果不理想，不能作为可靠模型
```

Google colab

```
https://colab.research.google.com/drive/1yG01IFE8tJU_E59iqEL0t2iFRyLaaTYA?usp=sharing
```

#### 数据

数据文件为pycode/input

选取中国四个城市的新冠肺炎确诊数据，城市为北京，上海，深圳，长春，其中选取上海，深圳，长春本轮疫情的爆发开始的第一天为第0天，然后推42天为我们数据预测天数，其中，为了排除历史数据干扰，我们将第0天的确诊病例归0。由于在此模型开发完成后，北京疫情仍然处于发展阶段，所以我们选取的是从数据筛选最后一天往前推42天作为北京的疫情数据。

使用数据的时候，我们以前12天的数据为基础，模型给出第十三天的确诊预测值。

#### 模型

模型文件为pycode/Transfromer.ipynb

采用ConvTrans模型，预测未来

机器学习对比文件

pycode/regression.py

LSTM对比文件

pycode/LSTM.py

#### 结论

Transformer在模型拟合上具有极大的优势，在如此小的数据量的情况下拟合效果仍然比较优秀

![img (6)](src\img (6).png)

相比于老的机器学习模型，我们可以看出Loss是明显占据优势的

![img (7)](src\img (7).png)

对比其他模型如LSTM模型，则仍然展现了自己的优势(对比数据均为深圳数据)

![img (5)](src\img (5).png)

在绘图可视化上，我们可以看出拟合度的情况

![img (4)](src\img (4).png)

![img (1)](src\img (1).png)

![img (2)](src\img (2).png)

![img (3)](src\img (3).png)

#### 结论

我们认为，transformer在预测新冠肺炎的情况下具有较大优势，整体上的损失情况均可以战胜一些其他的模型，值得我们进一步深入研究。

我们认为，未来研究中，可以增加的维度如下：

城市人口，迁入迁出，机动车日均流量，城市天气数据等

同时增加更多的数据，有期望可以将模型效果进一步提升

