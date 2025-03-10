
# Experiments

test:
```commandline
python test_detection.py --gpu 1 --manip ./data/IDT-CocoGlide-manip.txt --auth ./data/IDT-CocoGlide-auth.txt
```

train (phase 1):
```commandline
python3 ec_train.py --exp ./experiments/ec_example.yaml --gpu 1
```

train (phase 2):
```commandline
python3 ec_train_phase2.py --exp ./experiments/ec_example_phase2.yaml --gpu 1 --ckpt ckpt/ec_example/best_val_loss.pth 
```

## Model Detection (early_fusion):

| Datasets                                                                         | AUC    | bACC   | Time (CPU/GPU) (sec) | Count imgs | TP/TN/FP/FN     | (Manip) P/R/F1 | (not manip) P/R/F1 | F1 mean |
|----------------------------------------------------------------------------------|--------|--------|----------------------|------------|-----------------|----------------|--------------------|---------|
| CocoGlide                                                                        | 0.7547 | 0.684  |                      | 1024       | 253/448/64/259  | 0.79/0.49/0.61 | 0.63/0.87/0.73     | 0.67    |
| CocoGlide (после скейлинг (downscale-upscale) (бикубическая на 1.2 коэффициент)) | 0.5561 | 0.542  | 0.3870/0.037         | 1024       | 259/286/226/243 | 0.54/.52/0.53  | 0.54/0.55/0.54     | 0.53    |
| CocoGlide (сжатые JPEG - степень сжатия рандомная от 50% до 80%)                 | 0.6566 | 0.5917 | 0.4111               | 1024       | 191/415/97/321  | 0.66/0.37/0.47 | 0.56/0.81/0.66     | 0.56    |
| CocoGlide (both - сжатие + скейлинг                                              | 0.5294 | 0.5214 | 0.3575/0.0347        | 1024       | 277/257/255/235 | 0.52/0.54/0.53 | 0.52/0.50/0.51     | 0.52    |
| DSO-1                                                                            | 0.972  | 0.885  | ?/0.26               | 200        | 95/82/18/5      | 0.84/0.95/0.89 | 0.94/0.82/0.87     | 0.88    |
| DSO-1  (compress)                                                                | 0.767  | 0.665  | ?/0.26               | 200        | 66/67/33/34     | 0.66/0.66/0.66 | 0.66/0.67/0.66     | 0.66    |
| DSO-1  (resize)                                                                  | 0.884  | 0.8    | ?/0.26               | 200        | 72/88/12/28     | 0.85/0.72/0.78 | 0.76/0.88/0.81     | 0.79    |
| DSO-1 (both)                                                                     | 0.735  | 0.655  | ?/0.26               | 200        | 42/89/11/58     | 0.79/0.42/0.54 | 0.60/0.89/0.72     | 0.63    |

VK: приводит к качеству к (1024 по высоте для вертикальных фото, и 1280 по ширине для горизонтальных. + сжатие фото
Telegram: сжимает до 50% качества

Наборы :
* [CocoGlide](https://github.com/grip-unina/TruFor#cocoglide-dataset) 
* [DSO-1](https://recodbr.wordpress.com/code-n-data/#dso1_dsi1)



## Model Detection (late_fusion):

| Datasets                                                            | AUC | bACC  | Time (CPU/GPU) (sec) | Count images |
|---------------------------------------------------------------------|---|-------|----------------------|------|
| [CocoGlide](https://github.com/grip-unina/TruFor#cocoglide-dataset) | 0.76041 | 0.683 | 0.552 / 0.0774       | 1024 |

