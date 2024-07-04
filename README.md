<p align="center">
<h1 align="center"><strong>On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy</strong></h1>
<h3 align="center">ECCV 2024</h3>

<p align="center">
          <span class="author-block">
              <a href="https://letianhuang.github.io/">Letian Huang</a></span>&nbsp;&nbsp;&nbsp;&nbsp;
            <span class="author-block">
              <a href="https://scholar.google.com/citations?user=VmPQ6akAAAAJ&hl=zh-CN">Jiayang Bai</a></span>&nbsp;&nbsp;&nbsp;&nbsp;
            <span class="author-block">
              <a href="https://scholar.google.com.hk/citations?user=Sx4PQpQAAAAJ&hl=en">Jie Guo</a>
            </span>&nbsp;&nbsp;&nbsp;&nbsp;
            <span class="author-block">
              <a href="http://www.njumeta.com/liyq/">Yuanqi Li</a>
            </span>&nbsp;&nbsp;&nbsp;&nbsp;
            <span class="author-block">
              <a href="https://cs.nju.edu.cn/ywguo/index.htm">Yanwen Guo</a>
            </span>
    <br>
        Nanjing University
</p>

<div align="center">
    <a href=https://arxiv.org/abs/2402.00752><img src='https://img.shields.io/badge/arXiv-2402.00752-b31b1b.svg'></a>  
    <a href='https://letianhuang.github.io/op43dgs'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  
</div>

</p>

<p align="center">
<img src="https://letianhuang.github.io/assets/img/fisheye.png" width=100% height=100% 
class="center">
</p>

## News

**[2024.07.05]** Birthday of the repository <img class="emoji" title=":smile:" alt=":smile:" src="https://github.githubassets.com/images/icons/emoji/unicode/1f604.png" height="20" width="20">.

## TODO List
- [ ] Release the code.
- [ ] Release the submodule for panorama's rasterization.

## Motivation

<p align="center">
<img src="https://letianhuang.github.io/op43dgs/resources/motivation.png" width=100% height=100% 
class="center">
</p>

We derive the mathematical expectation of the projection error (Top left), visualize the graph of the error function under two distinct domains and analyze when this function takes extrema through methods of function optimization (Top right). We further derive the projection error function with respect to image coordinates and focal length through the coordinate transformation between image coordinates and polar coordinates and visualize this function, with the left-to-right sequence corresponding to the 3D-GS rendered images under long focal length, 3D-GS rendered images under short focal length, the error function under long focal length, and the error function under short focal length (Below).

## Pipeline

<p align="center">
<img src="https://letianhuang.github.io/op43dgs/resources/optimal_proj.png" width=100% height=100% 
class="center">
</p>

Illustration of the rendering pipeline for our Optimal Gaussian Splatting and the projection of 3D Gaussian Splatting. The blue box depicts the projection process of the original 3D-GS, which straightforwardly projects all Gaussians onto the same projection plane. In contrast, the red box illustrates our approach, where we project individual Gaussians onto corresponding tangent planes.

## Citation

```BibTeX
@article{huang2024erroranalysis3dgaussian,
    title={On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy},
    author={Letian Huang and Jiayang Bai and Jie Guo and Yuanqi Li and Yanwen Guo},
    journal={arXiv preprint arXiv:2402.00752},
    year={2024}
}
```
