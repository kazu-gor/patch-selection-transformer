# Patch Selection Transformer

## タスク

### Model

####  Swin-Transformer

- [x] Swin-TransformerをSwin-Transformer-Object-Detectionに変更
- [x] 各ステージの特徴量を取得
    
#### Transformer Decoder

- [x] Object Queryの数をSwin TransformerのPatch sizeに固定する
- [x] Object Queryの数をSwin-Transformerの各Window内のPatch sizeに固定
- [x] memoryとposのサイズが合わないバグ
- [ ] position embeddingを追加する
        
#### Patch selection part

- [ ] 各stageの出力をどう選ぶか決める（各ステージを足し合わせる？？）

### Dataset

- [x] targets sizeを元画像のpixel size数からObject Queryの数に変更
- [x] パッチ分割の教師データセットを作成

### Criterion

- [x] nn.BCEWithLogitsLoss()を適用
- [ ] パッチ内にごく僅かな石灰化しか含まれない場合の考慮
- [ ] 形状を考慮する

### Config
####  Dockerfile

- [x] mmcv, mmdet, openmimを追加
- [x] vimに対応
    
#### docker-compose.yml

- [x] docker-compose up --buildでビルドできるようにする
- [x] vimに対応