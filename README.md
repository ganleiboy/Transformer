# Transformer
学习Transformer相关知识时练手用的代码。

### 蒸馏学习
miniimagenet验证集精度：
| model | Parameters(M) | Accuracy | epoch | optimizer | alpha |
| ---- | -------------- | ------------ | ------------ | ------------ |------------ |
| resnet18 | 11.2 | 57% | 60 | SGD | / |
| studentnet（无蒸馏） | 0.4 | 24% | 60 | SGD | / |
| studentnet（hard蒸馏） | 0.4 | 24.7% | 60 | SGD | 0.5 |
| studentnet（soft蒸馏） | 0.4 | / | 60 | SGD | 0.5 |
| studentnet（soft蒸馏） | 0.4 | 24.1% | 60 | SGD | 1.0 |
| studentnet（soft蒸馏） | 0.4 | 25.0% | 100 | SGD | 1.0 |

#### 结论
- 如果teacher模型的精度很低，则没有什么作用，会导致学生模型效果很差。
- 使用蒸馏学习的话，需要的训练epoch更多。