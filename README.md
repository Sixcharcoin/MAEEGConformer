All files are summarised / refactored using LLMs.   
Some comments are my own, some comments are made by LLMs to enhance readability.   
Refer to maeegconformer_original.py for the original entire code.
--------------------------------------------------------------------------------   

# MAEEGConformer: EEG-based Image Category Classification
## タスク概要

- 入力: 17 チャンネル × 100 タイムステップの EEG 時系列データ
- 出力: 被験者が見ている画像のカテゴリ（5 クラス分類）
- 課題設定: ノイズの多さや被験者間の個人差を含む EEG 信号から、安定した特徴表現を学習し、高精度な分類を行うこと

本プロジェクトでは、CNN・Transformer・Conformer など複数の深層学習アーキテクチャを設計・比較し、最終的に Masked Autoencoder (MAE) による自己教師あり学習と Conformer を組み合わせたモデル **MAEEGConformer** を構築しました。

## データ前処理と拡張

### サンプル単位での標準化

被験者ごとのベースラインの違いを抑制するため、サンプルごとに全要素の平均・分散で Z-score 正規化を行いました。後に Subject Embedding（被験者埋め込み）を導入し、被験者ごとの特性をモデル側に学習させる方針をとったため、この標準化は一部冗長であった可能性があります。

### 確率的データ拡張

訓練データに対して、以下の 5 種類のデータ拡張を実装しました。

- ガウシアンノイズ付加
- タイムシフト
- チャンネルドロップアウト
- 時間方向の区間マスキング
- スケーリング

当初はこれらを同一サンプルに複数同時適用していましたが、性能が低下したため、「波形の原型を破壊し過ぎている」という仮説を立てました。そこで、1 サンプルあたり確率的に 1 種類のみ適用する方針に変更したところ、テスト精度が約 0.02 向上し、汎化性能の微弱な改善につながりました。

## モデルの主要コンポーネント

### Learnable Positional Encoding

固定のサイン・コサインによる位置エンコーディングではなく、系列中の「どの時点が重要か」をモデル自身が学習できる **学習可能な位置埋め込み** を採用しました。[1]

### Gated Attention Pooling

入力トークン群を重要度に応じて加重平均する Attention Pooling に対し、NeurIPS 2025 で評価された LLM（Qwen）のアーキテクチャに着想を得て、シグモイド関数によるゲート機構を追加しました。特定の時間ステップの重要度を強調する狙いでしたが、本データセットの規模では顕著な精度向上は確認できませんでした。データの複雑さやモデル容量とのトレードオフの可能性があります。

### Squeeze-and-Excitation (SE) Layer

EEGNetHybrid では、チャネル間の重要度を学習する Squeeze-and-Excitation (SE) Layer を導入しました。Transformer が主に時間方向の文脈を扱うのに対し、SE Layer は各電極チャンネルの空間的重要度を再重み付けする役割を担います。

### Conformer Block

Conformer は、局所的なパターン抽出に強い CNN と、大域的な依存関係を捉える Transformer を 1 ブロック内で組み合わせたアーキテクチャです。EEG のような時系列信号に対して、局所・大域の両方の特徴を同時に捉えることを意図しています。本実装では Transformer との親和性を高めるため、正規化層には LayerNorm を採用しました。

### Masked Autoencoder (MAE) と Subject Embedding

Apple の MAEEG を参考に、ラベルなし EEG データの一部をマスクし、残りのパッチから波形全体を復元させる自己教師あり学習を導入しました。

- 入力をパッチ化し、位置埋め込みと Subject Embedding（被験者 ID 埋め込み）を付与
- ランダムに約 50% のパッチをマスク
- 残りのパッチから全パッチを再構成するタスクでエンコーダを事前学習
- 学習済みエンコーダを分類タスクに転移し、ファインチューニング

これにより、ラベルに依存しない EEG の構造的な特徴表現を獲得しやすくなりました。

## モデルバリエーションと性能

本プロジェクトでは、以下のモデル群を設計・比較しました（Acc は検証データにおける概算値）。

### EEGNetHybrid (Acc: 約 0.50)

- EEGNet ベースの CNN で局所特徴を抽出し、その出力を Transformer Encoder に入力する構成
- 原論文の畳み込み順序「Temporal → Spatial」を「Spatial → Temporal」に反転したところ、経験的に精度が約 0.02 向上したため、この構成を採用

### EEGConformer (Acc: 約 0.51)

- EEGNetHybrid の Transformer 部分を 2 層の Conformer ブロックに置換
- 時間構造のモデリング能力を強化することで、わずかに精度を改善

### MAEEG with Subject Embedding (Acc: 約 0.52)

- MAE を用いて EEG 波形の自己教師あり事前学習を実施
- エンコーダに Subject Embedding を導入し、被験者固有の特徴を明示的に埋め込むことで、個人差を表現可能な潜在表現を学習
- 事前学習したエンコーダを分類タスクに転移し、単独モデルとして EEGConformer よりも高い精度を達成

### MAEEGConformer (Acc: 約 0.53)

- MAEEG の Transformer Encoder を 4 層の Conformer ブロックに全面置換した最上位モデル
- MAE による表現学習と Conformer の時系列モデリング能力を組み合わせることで、単独モデルとして最高の検証スコアを記録

## 推論時の工夫と最終スコア

テストデータに対する最終的なスコアの底上げのため、以下の推論手法を組み合わせました。

### Test-Time Augmentation (TTA)

推論時に各サンプルに対して微小なノイズ付加やシフトを行った 4 パターンの入力を生成し、それぞれの出力ロジットを平均することで予測の分散を抑えました。

### 重み付きアンサンブル

- MAEEGConformer
- MAEEG
- EEGConformer
- EEGNetHybrid

上記 4 モデルの出力ロジットに対し、各モデルの検証 Accuracy に比例する重みを付与して加重平均を行い、最終予測確率を算出しました。

これらの工夫により、最終的なリーダーボードスコア **0.53199** を達成しました。

## 反省、考察と今後の展望   
   
### 能力の不足 

論文を正確に理解し実装する能力が根本的に足りておらず、たとえばMAEEGのマスキングをかけるタイミングを誤解し生データに適用してしまったりしていました。うまくAIとも付き合いつつ正確な実装をする能力を養っていくべきだと反省しました。

### ハイパーパラメータ探索の不足

アーキテクチャ設計と実装に時間を割いた結果、Optuna などを用いた体系的なハイパーパラメータ探索（学習率、Mixup の \(\alpha\) 値、Weight Decay など）が十分に行えていません。今後は、自動探索ツールを活用して探索空間を広くカバーすることで、さらなる精度向上が見込めます。

### コード設計と可読性

実験を高速に回すことを優先した結果、類似するクラスや関数が重複定義され、コードベースが肥大化しました。DRY (Don't Repeat Yourself) の原則とモジュール性をより意識し、再利用しやすいコンポーネント設計へとリファクタリングする余地があります。

### マルチモーダル拡張の可能性

本タスクでは EEG 信号のみを入力としましたが、分類対象となる画像データ自体の利用も許可されていました。将来的には、CLIP などで抽出した視覚特徴と EEG 特徴を同一空間にマッピングするマルチモーダル学習を導入することで、「どのような画像を見ているか」という意味情報をより直接的に反映させることができると考えています。

## 参考文献

 1. Positional Encoding: https://apxml.com/courses/foundations-transformers-architecture/chapter-4-positional-encoding-embedding-layer/comparing-positional-encodings
 2. Squeeze-and-Excitation Networks: https://arxiv.org/abs/1709.01507  
 3. EEGNet: https://arxiv.org/abs/1611.08024  
 4. MAEEG (Masked Auto-Encoder for EEG Representation Learning): https://machinelearning.apple.com/research/masked-auto-encoder
