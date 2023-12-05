![example workflow](https://github.com/kaseris/skelcast/actions/workflows/ci.yml/badge.svg)
# skelcast
<img src="assets/skelcast-logo.png" height=400></img>

Skeletal join forecasting
# How to install

Create a new python `venv` environment
```bash
python3 -m venv myenv
```

Install the package

```bash
pip install -r requirements.txt
pip install --editable .
```

# Docker
If you are familiar with Docker you can build and run the image

```bash
docker build -t [YOUR-IMAGE-NAME] .
```

*NOTE 1*: Due to the fact that during the training process there is a lot of reading and writing to the filesystem, you will have to add the following volumes, so that they are accessible from your host machine. Here's how you can run it:

*NOTE 2*: As of now, only `train` mode is supported.

```bash
docker run --gpus all \
-v ~path/to/your/data/dir:/usr/src/app/dir \
-v ~path/to/checkpoints_forecast:/usr/src/app/checkpoints_forecast \
-p 6006:6006 [YOUR-IMAGE-NAME] [train/infer] \
--data_dir /usr/src/app/dir \
--checkpoint_dir /usr/src/app/checkpoints_forecast \
--config configs/lstm_regressor_1024x1024.yaml
```