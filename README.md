# PyTorch & OpenCV を用いた、霧箱における飛跡のリアルタイム分類

webcamera で霧箱内の映像を取得し、背景差分で飛跡を抽出したら、PyTorch で深層学習にかけて α 線と宇宙線の分類を行う
<br><br>

<dl>

## <dt>実行環境</dt>
- **Web Camera**: Buffalo BSW500M
- **PC**: NUC13ANKi5  (CPU: 13th Gen Intel(R) Core(TM) i5-1340P / 32GB)
- Pythonの実行環境はrequirements.txtを参照。
- venvの仮想環境を作成後、有効化してpip installを実行。

```.sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## <dt>使い方</dt>

### <dd>Step1: 学習データの作成

-  **take_photo.py** を実行し、学習に使う画像を撮影する。
- 飛跡の見えていない状態で背景を撮影したら、飛跡の撮影を行うと飛跡部分を切り取った画像が **image_data/tmp** に保存される。
- グレースケールした後で、背景画像との差分の絶対値にメディアンフィルターを適用し、ノイズ除去を行なった状態で二値化し、マスクをかける。このとき、Hough 変換によって直線検出されたものの一部が緑の枠に囲まれて表示される。
- 液体窒素の霧箱を使う場合、背景画像は定期的に更新されるがペルチェ霧箱では背景画像を最初に撮影する。ペルチェ霧箱を使う場合はbranchを切り替えるように。

```.sh
git checkout peltie
```

<br><br>

</dd>

### <dd>Step2: データのラベリング

- 撮影した画像データに対して **label_img.py** を実行し、ラベル付けを行う。

- ラベル付けされたデータは  **image_data/all/alpha** (cosmic) に保存される。<br><br>

</dd>

### <dd>Step3: PyTorchで学習

- **train_net.py** を実行し、作ったデータをもとに学習を行う。

- 学習済みのパラメーターは  **trained_net.ckpt** に保存され、途中から学習を再開できる。<br><br>

</dd>

### <dd>Step4: リアルタイム分類

  - **realtime_classification.py** を実行し、**trained_net.ckpt** での学習結果に基づいてリアルタイムでの分類を行う。

- フリーズのおそれがあるため、実行に使用するコア数（デフォルト4）の変更を推奨
- バッチサイズ (一度に検出する物体の数) の上限数によってはフリーズするおそれがあるので注意。
<br><br>


</dd>

## <dt>学習モデル</dt>

<img width="908" alt="スクリーンショット 2025-01-10 22 04 05" src="https://github.com/user-attachments/assets/47ed5bf9-ace7-4168-985c-c818232c40d6" />




## <dt>参考</dt>

作成にあたり、以下の文献を参考にした。

- Raspberry 3 B+ ＆ PyTorchの深層学習で、カメラ映像内の複数物体をリアルタイム分類
  
  https://qiita.com/AoChoco/items/a09b446460d95d5c9503

- 霧箱を用いた放射線の測定

  https://osksn2.hep.sci.osaka-u.ac.jp/theses/soturon/sotsuron2021-1.pdf

</dl>
