# DCGAN-with-facedata
This repository Investigates DCGAN using facedata. Serves as a personal cautionary tale when working with GANS.

## Some important stuff I learnt
 - Please make sure you normalize and rescale the training data properly in the order described. Normalize/standardise first(zero mean unit variance) then rescale. Doing this the reverse way may be a cause of headache. 
 - Kernel sizes and strides are hugely important. Please, please ensure that your kernel sizes at least learn something and are big(or small) enough to capture reasonable data from the input. 
 - Play around with various loss functions. If one loss function 'isnt working for you', please feel free to try other loss functions available that are mostly used for these systems.  
 - Hyperparameters are **massively** important in GANS i.e learning rates, kernel sizes, strides, padding, what you normalize/standardize it by even the rescale min/max. Keep adjusting these parameters appropriately until you see the model is learning something.
If you have poor hyperparameters just forget about model convergence.
 - Monitor gradient flow both in generator and discriminator. Especially the discriminator. A good gradient flow in the discriminator will greatly influence generator performance.
 - Investigate general model structure using torch.summary especially backward/forward pass size. If the size seems abit off compared to what you se online then something somewhere is wrong.
From my experience its mostly, well, hyperparameters(latent space size, kernel size and stride length).
 - Adding noise and dropouts to the network is helpful. But before you add some spice to your network please ensure your hyperparameters are good and gradients are flowing.
 - Optimizer is also one of the major factors. Adam optimizer is seen to work best.
 - Low learning rates help but to some degree. Requires trial and error to determine what learning rate is best.

Hopefully these points will help you save alot of time when you have trouble training a GAN :)
