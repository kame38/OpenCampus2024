## RaspberryPi 3 B+ ＆ PyTorch 深層学習を用いた、霧箱における飛跡のリアルタイム分類

picamera で霧箱内の映像を取得し、背景差分で飛跡を抽出したら、PyTorch で深層学習にかけて α 線と宇宙線の分類を行う

### 実行環境

### 使い方

<dl>

<dt>#### Step1: 学習データの作成</dt>

<dd>
RaspberryPi 上で take_photo.py を実行し、学習に使う画像を撮影する。

- 飛跡の見えていない状態で背景を撮影したら、飛跡の撮影を行うと飛跡部分を切り取った画像が image_data/tmp に保存される。
- 背景画像との差分の絶対値を二値化しマスクをかけたものに対し、Hough 変換によって直線検出したものが緑の枠に囲まれて表示されている。
</dd>

<dt>#### Step2: データのラベリング</dt>

<dd>
撮影した画像データに対して label_img.py を実行し、ラベル付けを行う。

- ラベル付けされたデータは image_data/all/alpha(cosmic)に保存される。
</dd>

</dl>

### 参考

コード作成をするにあたり、以下の文献を参考にしている。

-「Raspberry Pi 3 B+ ＆ PyTorch の深層学習で、カメラ映像内の複数物体をリアルタイム分類」

https://qiita.com/AoChoco/items/a09b446460d95d5c9503

-「霧箱を用いた放射線の測定」

https://osksn2.hep.sci.osaka-u.ac.jp/theses/soturon/sotsuron2021-1.pdf
