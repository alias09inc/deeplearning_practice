# aliAsの深層学習アウトプット場
ここでは私が深層学習について学んだ内容をアウトプットするリポジトリとして使用しています。

## 実行環境について
-   Windows11 + WSL(Ubuntu22.04)
-   Ryzen 7 4800HS
-   GeForce GTX 1660 Ti Max-Q
-   Python: 3.10.6
-   Pytorch: 2.0.1
-   CUDA Version: 12.2
-   NVCC: 11.5

以上が主要な環境となっています。pythonの詳しい拡張機能については`requirements.txt`を参照してください。

## CNNについての学習
畳み込みニューラルネットワークに関する学習成果物はCNNディレクトリにあります。現状手書き数字の認識であるMNISTとファッション商品の認識であるFashion-MNISTの2種類を対象にPytorchを用いて実装してあります。今後、学習モデルの保存や読み込みに関する部分も実装予定です。