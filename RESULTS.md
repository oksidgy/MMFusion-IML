
# Experiments


## Model Detection (early_fusion):

| Datasets                                                                         | AUC    | bACC   | Time (CPU/GPU) (sec) | Count images |
|----------------------------------------------------------------------------------|--------|--------|----------------------|------|
| [CocoGlide](https://github.com/grip-unina/TruFor#cocoglide-dataset)              | 0.7547 | 0.684  |                      | 1024 |
| CocoGlide (после скейлинг (downscale-upscale) (бикубическая на 1.2 коэффициент)) | 0.5561 | 0.539  | 0.3870               | 1024 |
| CocoGlide (сжатые JPEG - степень сжатия рандомная от 50% до 80%)                 | 0.6621 | 0.589  | 0.4111               | 1024 |
| CocoGlide (both - сжатие + скейлинг                                              | 0.5468 | 0.5283 | 0.3575/0.0347        | 1024 |


VK: приводит к качеству к (1024 по высоте для вертикальных фото, и 1280 по ширине для горизонтальных. + сжатие фото
Telegram: сжимает до 50% качества


## Model Detection (late_fusion):

| Datasets                                                            | AUC | bACC  | Time (CPU/GPU) (sec) | Count images |
|---------------------------------------------------------------------|---|-------|----------------------|------|
| [CocoGlide](https://github.com/grip-unina/TruFor#cocoglide-dataset) | 0.76041 | 0.683 | 0.552 / 0.0774       | 1024 |

