## PytorchでSoftmax関数を定義しない理由

PyTorch では計算を高速化させるために、Softmax 関数と対数変換を同時に行う関数である F.log_softmax が用意されています。`Softmax => Log` の計算がそれぞれ行うと速度として遅いこと、そして計算後の値が安定しないといった理由で、それぞれを別々に計算するのではなく一度にまとめて計算する方が優れているようです。そして、PyTorch で用意されている `F.cross_entropy` では内部の計算に `F.log_softmax` が使用されています。したがって、事前に Softmax 関数の計算を行う必要がありません。そのため、モデルの定義の部分では Softmax 関数を設けていませんでした。

[参照: キカガク](https://free.kikagaku.ai/tutorial/basic_of_deep_learning/learn/pytorch_basic)

## Pytorch上での画像の扱いについて
Tensor imagesを基本的に扱う。データ型がfloatの場合はデータ範囲が`[0,1)`となっていて、整数型の場合は`[0, MAX_DTYPE]`となっている。