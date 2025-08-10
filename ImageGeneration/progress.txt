## Was fascinated by Image Generation by AI, so decided to give it a try myself

# Model V1
    Description - Created a basic DCGAN model learning on a dataset of anime character face images
    Learning details about how that mathematics behind this works and how to make it better
    Added residual connections and logics that I learnt while learning Image Classification

# Model V2
    Description - While figuring out to make this better, learnt a lot more theory like
    hinge loss and other stuff and implemented that in this version (some improvement but not a lot)

# Model V3
    Desciption - To make the model found out about WGANs and WGAN-GPs so put it to the test.
    Normalization layers play a big role in this so, explored and triedd all my options to find GroupNorm working the best
    Read the fascinating math behind GradientPenalty(GP) in WGAN-GP models

# Model V4
    Description - Model2 + using noise to prevent discriminator overpowering and providing diversity to the generator
    Basically keeping the ResNet structure intact in the discriminator and improving the generation by adding random gaussian noise
    in the training loops and removing it when saving images for per epoch checks

# Model V5
    Desciption - Finally, after weeks of efforts, understood and implemented the StyleGAN research paper published by NVIDIA
    The StyleGAN paper worked majorly on the Generator architecture improvement using a learnt style vector to inject styles into the images
    Also, been using ConvTranspose layers for generation uptil this model, used the (upscale + conv) combo in this one (Worked out pretty well!)
    Not to forget, the most important part of the model Adaptive Instance Normalization layers (fascinating concepts!).
    Implemented Multilayer Perceptrons, HingeLosses, Adaptive Instance Normalization, Mini Batch Standard Dev
    and a lot more concepts that I hadnt even heard of before.
    Tried out ResNet style-discriminator.

