Adversarial Attack with min/max L2-norms
==============

The objectives of <code>adversarial-min-max</code> are:
1. Create adversarial images that can fool a pre-trained classifier.  
2. Minimise or maximise the L2-norm scores of each image, depending on the configuration. 

# Requirements
- numpy
- pillow
- pytorch

# Adversarial attacks
The algorithms are re-implementations of Deepfool by Moosavi-Dezfooli et al. (2016) and Iterative-Fast Gradient Sign Method by Goodfellow et al. (2014). 

# Model and data
- Please use your trained model for inference when conducting adversarial attack; the model mentioned in this repository is a black box for demonstration. 
- Ditto input images.  

# Examples
Below are some example showing how to run the <code>main.py</code> to generate the minimum and maximum L2 adversarial images.

<code>$ python main.py --py-version 3.7 --mode min --data-path ../data --model-path ../model/model.pt --save t --output-path ../output/Deepfool</code>

<code>$ python main.py --py-version 3.7 --mode max --data-path ../data --model-path ../model/model.pt --save t --output-path ../output/IGSM</code>

![Original 77 screenshot](/data/artifacts/77.png?raw=true)
![Deepfool 77 screenshot](/output/Deepfool/77.png?raw=true)
![IGSM 77 screenshot](/output/IGSM/77.png?raw=true)

![Original 3578 screenshot](/data/normal_regions/3578.png?raw=true)
![Deepfool 3578 screenshot](/output/Deepfool/3578.png?raw=true)
![IGSM 3578 screenshot](/output/IGSM/3578.png?raw=true)
