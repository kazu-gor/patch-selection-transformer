## Patch Selection Transformer

### Implementation

- [x] Transformer Decoder
    - [ ] Object Queryの数をSwin Transformerの各Window内のPatch sizeに固定する
    - [x] Object Queryの数をSwin TransformerのPatch sizeに固定する
- [x] Patch selection part
    - [x] selection partは後処理とする

### Bug

- Decoder
    - [ ] memoryとposのサイズが合わないバグ