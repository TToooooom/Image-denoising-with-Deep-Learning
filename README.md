
# Image-denoising-with-Deep-Learning

基于深度学习的图像去噪应用，尝试了 CNN、AutoEncoder、U-Net 与 U-Net Residual 四种网络结构，并加入传统高斯滤波作为 baseline 对比。本项目为深度学习课程设计作品。

Deep learning based image denoising application. This project explores CNN, AutoEncoder, U-Net, and U-Net Residual architectures, and also provides traditional Gaussian filtering as a baseline. This is a course project for deep learning.

由于 GitHub 文件大小限制，`checkpoints/` 中的模型权重文件未上传至 branch 中。训练后模型文件会自动保存至 `checkpoints/` 文件夹，也可以[直接下载模型文件](https://github.com/TToooooom/Image-denoising-with-Deep-Learning/releases/download/Checkpoints/checkpoints.zip)

Due to GitHub file size limitations, model checkpoint files in `checkpoints/` are not uploaded to the branch. After training, model files will be automatically saved to the `checkpoints/` folder. You can also [download pretrained model files](https://github.com/TToooooom/Image-denoising-with-Deep-Learning/releases/download/Checkpoints/checkpoints.zip)

---

## 1. 模型说明 / Models

| Model | 说明 | Description |
|---|---|---|
| CNN | 简单卷积去噪 | A simple convolutional denoising baseline |
| AutoEncoder | 编码器-解码器结构，通过压缩表示进行图像恢复 | Encoder-decoder model for image restoration |
| U-Net | 编码器-解码器结构，使用 skip connection 保留图像细节 | Encoder-decoder model with skip connections |
| U-Net Residual | 预测噪声残差，再从输入图像中减去噪声 | Predicts the noise residual and subtracts it from the noisy input |
| Gaussian Filter | 传统高斯滤波方法，不需要训练 | Traditional Gaussian filtering baseline without training |

---

## 2. 配置 / Setup

建议使用 Python 3.9 或以上版本。

Python 3.9 or above is recommended.

安装依赖并准备数据集

Install dependencies and prepara dataset

本项目使用本地图像文件夹作为数据源，请将训练图片放入：

This project uses a local image folder as the dataset. Please put training images into:

```
data/
```

---

## 3. 使用说明 / Application Usage

运行交互式应用：

Run the interactive application

```
python app.py
```

打开浏览器访问127.0.0.1:7860

Open the browser and visit 127.0.0.1:7860

左侧可以上传待处理图像并配置参数，右侧会输出处理结果。

The left panel allows users to upload an image and configure parameters. The right panel displays the output results.

### 3.1. 运行模式 / Running Modes

APP支持两种运行模式：

The APP supports two running modes：

#### 模式一：对干净图像加噪后再去噪
Mode 1: Add noise to a clean image and then denoise

该模式会先对上传的干净图像加入噪声，然后进行去噪。右侧会输出：

This mode first adds synthetic noise to the uploaded clean image and then performs denoising. The right panel outputs:

```
Noisy image
Denoised image
Clean-Noisy-Denoised comparison
PSNR and SSIM metrics
```

由于该模式下有干净图像作为 ground truth，因此可以计算 PSNR 和 SSIM。

Since the clean image is available as the ground truth, PSNR and SSIM can be computed.

#### 模式二：直接对上传图像去噪
Mode 2: Directly denoise the uploaded image

该模式直接对上传图像进行去噪。右侧会输出：

This mode directly denoises the uploaded image. The right panel outputs:

```
Denoised image
Input-Denoised comparison

```

### 3.2. 去噪方法 / Denoising Methods

APP 支持两种去噪方法：

The APP supports two denoising methods:

#### 深度学习方法 / Deep Learning Method

选择已经训练好的 checkpoint 进行去噪。可使用的模型包括 CNN、AutoEncoder、U-Net 和 U-Net Residual。

Select a trained checkpoint for denoising. Available models include CNN, AutoEncoder, U-Net, and U-Net Residual.

#### 传统高斯滤波方法 / Traditional Gaussian Filtering

无需模型训练，可直接调整高斯滤波半径进行传统滤波去噪。

No training is required. Users can adjust the Gaussian filter radius for traditional denoising.

---

## 4. 训练与评估说明

本项目支持两种训练方式：

This project supports two training modes:

```
1. train.py: train one specified model
2. run_experiments.py: train multiple models sequentially
```

### 4.1. 单模型训练 / Training a Single Model

使用train.py可以训练指定模型(CNN, Autoencoder, UNet, UNet-Residual)。

Use train.py to train one specified model(CNN, Autoencoder, UNet, UNet-Residual).

示例：训练U-Net：

Example：training U-Net

```
python train.py --model_name unet --image_dir "./data/voc_images" --noise_type gaussian --noise_level 0.1 --image_size 256 --resize_size 288 --batch_size 4 --epochs 20 --loss_type charbonnier_mse --charbonnier_weight 0.8 --mse_weight 0.2 --lr 0.001 --weight_decay 0.00001 --num_workers 0 --seed 42 --device auto --val_every 1
```

### 4.2. 多模型批量训练 / Training Multiple Models

使用 run_experiments.py 可以依次训练多个模型。目前脚本默认训练：

Use run_experiments.py to train multiple models sequentially. By default, it trains:

```
cnn
autoencoder
unet
unet_residual
```

示例：训练四个模型，并在训练完成后自动评估：

Example: train all four models and automatically evaluate them after training:

```
python run_experiments.py --image_dir "./data/voc_images" --noise_type gaussian --noise_level 0.1 --image_size 256 --resize_size 288 --batch_size 4 --eval_batch_size 4 --epochs 15 --loss_type charbonnier_mse --charbonnier_weight 0.8 --mse_weight 0.2 --lr 0.001 --weight_decay 0.00001 --num_workers 0 --seed 42 --device auto --val_every 1 --run_eval --num_save_images 10
```

该命令将依次完成：

This command will sequentially:

```
Train CNN
Train AutoEncoder
Train U-Net
Train U-Net Residual
Evaluate each trained model
Save checkpoints, training logs, curves, metrics, and visual results
```

### 4.3. 主要参数 / Main Parameters

| Parameter | 说明 | Description |
|---|---|---|
| --image_dir | 图像数据集路径 | Path to the image dataset |
| --model_name | 指定训练模型，仅用于 train.py | Model name, only used in train.py |
| --noise_type | 噪声类型，支持 gaussian 和 salt_pepper | Noise type, supporting gaussian and salt_pepper |
| --noise_level | 噪声强度；Gaussian 下为标准差 sigma | Noise level; standard deviation sigma for Gaussian noise |
| --image_size | 训练输入 patch 大小，例如 128 或 256 | Training patch size, such as 128 or 256 |
| --resize_size | 随机裁剪前的缩放尺寸 | Resize size before random crop |
| --bath_size | 训练 batch size | Training batch size |
| --eval_batch_size | 评估 batch size | Evaluation batch size |
| --epochs | 训练轮数 | Number of training epochs |
| --loss_type | 损失函数类型，推荐 charbonnier_mse | Loss type, charbonnier_mse is recommended |
| --charbonnier_weight | Charbonnier Loss 权重 | Weight of Charbonnier Loss |
| --mse_weight | MSE Loss 权重 | Weight of MSE Loss |
| --lr | 学习率 | Learning rate |
| --weight_decay | 权重衰减 | Weight decay |
| --num_workers | DataLoader 子进程数，Windows 下建议为 0 | Number of DataLoader workers; 0 is recommended on Windows |
| --seed | 随机种子 | Random seed |
| --device | 训练设备，auto 表示自动选择 GPU 或 CPU | Training device; auto selects GPU or CPU automatically |
| --val_every | 每隔多少轮验证一次 | Validation frequency |
| --run_eval | 训练完成后是否自动评估 | Whether to run evaluation after training |
| --num_save_images | 评估时保存的可视化图片数量 | Number of visualization images saved during evaluation |

---

## 5. 推荐命令 / Recommended Command

推荐使用以下命令训练四个模型并自动评估：

The following command is recommended for training and evaluating all four models:

```
python run_experiments.py --image_dir "./data/voc_images" --noise_type gaussian --noise_level 0.1 --image_size 256 --resize_size 288 --batch_size 4 --eval_batch_size 4 --epochs 15 --loss_type charbonnier_mse --charbonnier_weight 0.8 --mse_weight 0.2 --lr 0.001 --weight_decay 0.00001 --num_workers 0 --seed 42 --device auto --val_every 1 --run_eval --num_save_images 10
```

该设置在训练成本和视觉效果之间较为平衡。

This setting provides a reasonable balance between training cost and visual quality.
