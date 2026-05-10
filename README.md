# Image-denoising-with-Deep-Learning
基于深度学习的图像去噪应用，尝试了 CNN、UNet 与 Autoencoder 三种网络结构（课程设计作品）。
Deep learning based image denoising – explored CNN, UNet &amp; Autoencoder architectures (course project).
由于Github上传文件大小限制，Checkpoints未上传至Brunch中。训练后模型文件自动保存至checkpoints文件夹或[直接下载模型文件](https://github.com/TToooooom/Image-denoising-with-Deep-Learning/releases/download/Checkpoints/checkpoints.zip)。

##使用说明

运行app.py，打开浏览器进入127.0.0.1:7860。左侧可以上传待处理图像并进行配置，右侧输出结果。

可用两种处理方式：

对干净图像加噪后再去噪，右侧输出加噪后图像、去噪后图像、Clean-Noisy-Denoised图像对比以及SSIM和PSNR指标。

直接对图像去噪：右侧输出去噪后图像以及去噪前后图像对比。

可用两种去噪方法：

深度学习方法：选择要使用的去噪模型并配置。

传统高斯滤波方法：可自行调整高斯滤波半径。

##训练与评估说明
进入项目根目录，运行命令以指定参数自动进行训练和评估。
