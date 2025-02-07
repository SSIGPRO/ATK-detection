# ATK-detection
attack detection

## Data sets creation

To create datasets of attacked images
```sh
xp_atk_cv.py
```

To create datasets with attacked and non-attacked images for training the discriminator
```sh
xp_discriminator_dataset.py
```

## Detectors

Tuning the detectors and evaluating the detectors given a configuration from tuning

```sh
xp_tuning_detectors.py
xp_detectors.py
```

## Discriminator
Tuning the discriminator and evaluating it given a configuration

```sh
xp_tuning_discriminator.py
xp_discriminator.py
```
