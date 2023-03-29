# TRCAN

TRCAN (Terrain Residual Channel Attention Networks) is a state-of-the-art terrain super-resolution model proposed in the paper "Adaptive & Multi-Resolution Procedural Infinite Terrain Generation with Diffusion Models and Perlin Noise".


<p align="center">
<img src=".\figures\terrain_sr.png" width="500" /><br/>
Terrain enhancement where (a) is enhanced to (b).  
</p>


It was modified and tuned for terrain super-resolution based on the natural image super-resolution model RCAN. The code structure is kept simple, with three main files for data preparation, training, and testing: prepare_tiles.py, train.py, and test.py.

## Data Preparation
To use the `prepare_tiles.py` file, you will first need to download the dataset from https://gitrepos.virvig.eu/oargudo/fcn-terrains. This file is used to break a terrain tile into patches, and the tile size is set to 256 by default in our implementation. Futher details are given in https://gitrepos.virvig.eu/oargudo/fcn-terrains.

## Training
To train the model, use the `train.py` file. Before running the file, update the paths where indicated in the code. The model weights will be saved after training is complete.

Compared to RCAN, minor modifications were made to the model to adapt it to terrain data. These modifications include changing the model head to accommodate the low-resolution/high-resolution correspondence in terrain tile preparation, and fine-tuning the model parameters.

<p align="center">
<img src="./figures/trcan.png" width="700" /><br/>
Architectural diagram of the proposed terrain super-resolution model TRCAN.
</p>

## Testing
To test the model on the test set, use the `test.py` file. Before running the file, update the paths where indicated in the code. The post-processing technique proposed and used in this file may take some time to complete. If you observe diminishing returns in PSNR gains vs time taken to post-process, you can change the limit in the code.

<p align="center">
<img src="./figures/sr_loss.png" width="400" /><br/>
Effect of stride on the RMSE.
</p>

## Trained Model Weights
The trained model weights are saved in the file `weights.pth`.

## Dependencies
This project uses standard libraries including PyTorch, Numpy, and PIL.


## Links

Project page link: https://3dcomputervision.github.io/publications/inf_terrain_generation.html

Video link: https://www.youtube.com/watch?v=6Uz6m4piXYI

Paper link: https://3dcomputervision.github.io/assets/pdfs/inf_terrain_generation.pdf

## BibTeX

```
@inbook{10.1145/3571600.3571657
  author = {Jain, Aryamaan and Sharma, Avinash and Rajan, K S},
  title = {Adaptive &amp; Multi-Resolution Procedural Infinite Terrain Generation with Diffusion Models and Perlin Noise},
  year = {2022},
  isbn = {9781450398220},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3571600.3571657},
  booktitle = {Proceedings of the Thirteenth Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP'22), December 8--10, 2022, Gandhinagar, India},
  articleno = {57},
  numpages = {9}
}
```
