# CV课程大作业

# 可复现性:

> 需要修改的只有 [config.yaml](config.yaml)， 为了结果的可重现性，已经设置了固定的随机种子。

`文件结构`

---<data_root>

-----training

----------1st_manual

----------images

-----------mask

-----test

----------1st_manual

----------2nd_manual

----------images

-----------mask

# 环境:

- python3.8
- pytorch==1.10.0

```python3
pip3 install -r requirement.txt
```

# 如何选择不同的网络块 :cake:：

> 不同网络块都写成一个类了,V-Net的参数中有一个参数供网络块传参使用, 只需要将对应的网络块传进去就行
