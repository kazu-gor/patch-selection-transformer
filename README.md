## Patch Selection Transformer

### 2023/07/11

- Transformer Decoder
    - Object Queryの数をSwin TransformerのPatch sizeに固定する
- Patch selection part
    - selection partは後処理とする

### 2023/07/19

- Dockerfile
    - added: mmcv, mmdet, openmim
- docker-compose.yml
    - added: build
- Swin-Transformer
    - Swin-TransformerをSwin-Transformer-Object-Detectionに変更

### 残タスク

- [ ] Transformer Decoder
    - [ ] Object Queryの数をSwin-Transformerの各Window内のPatch sizeに固定する
- [ ] Dataset
    - [ ] targets sizeを元画像のpixel size数からObject Queryの数に変更
- [ ] Decoder
    - [ ] memoryとposのサイズが合わないバグ