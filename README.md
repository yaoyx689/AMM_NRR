# AMM_NRR
This repository includes the source code of the paper [Fast and Robust Non-Rigid Registration Using Accelerated Majorization-Minimization](https://arxiv.org/abs/2206.03410).

Authors: Yuxin Yao, [Bailin Deng](http://www.bdeng.me/), [Weiwei Xu](http://www.cad.zju.edu.cn/home/weiweixu/) and [Juyong Zhang](http://staff.ustc.edu.cn/~juyong/).

This code is protected under patent. It can be only used for research purposes. If you are interested in business purposes/for-profit use, please contact Juyong Zhang (the corresponding author, email: juyong@ustc.edu.cn).

## Dependencies
1. [OpenMesh](https://www.graphics.rwth-aachen.de/software/openmesh/)
2. [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)

## Compilation
The code is compiled using [CMake](https://cmake.org/) and tested on Ubuntu 16.04 (gcc5.4.0) and Ubuntu 20.04 (gcc9.4.0). 

Run `./Compile.sh` and an executable `AMM_NRR` will be generated.

## Usage
Running `./run.sh` can execute the test example.

The program is run with four input parameters:
```
$ ./AMM_NRR <srcFile> <tarFile> <outPath> <landmarkFile>
or 
$ ./AMM_NRR <srcFile> <tarFile> <outPath> <radius> <alpha> <beta>
```
1. `<srcFile>`: an input file storing the source mesh;

2. `<tarFile>`: an input file storing the target mesh or point cloud; 

3. `<outPath>`: an output file storing the path of registered source mesh; 

4. `<landmarkFile>`: an landmark file (nx2 matrix, first column includes the indexes in source file, second column includes the indexes in target file, each row is a pair correspondences separated by space). `<landmarkFile>` can be ignored, our robust non-rigid registration method without landmarks will be used in this case.

5. `<radius>`: the sampling radius of deformation graph. 

6. `<alpha>`: the weight parameter of `regularization term`.

7. `<beta>`: the weight parameter of `rotation term`. 

### Notes

This code supports non-rigid registration from a triangle mesh to a mesh or a point cloud.

### Parameter choices
1. The weight parameters of `regularization term` and `rotation term` can be set in `paras.alpha` and `paras.beta` in `main.cpp` respectively or type in the command line. You can increase them to make the model more maintain the original characteristics, and decrease them to make deformed model closer to the target model. 
2. The sampling radius of deformation graph can be set in `paras.uni_sample_radio` in `main.cpp` or type in the command line. 
If you meet the error ''Error: Some points cannot be covered under the specified radius, please increase the radius'', you can
    - check if there are isolated vertices in the input source model. If yes, please preprocess the input source model so that there are no isolated vertices (Points not connected to any faces).
    - increase node sampling radius of deformation graph by setting "paras.uni_sample_radio" with a bigger value in "main.cpp". 
This is because the node sampling method is uniform. When the model points are not very uniform and the sampling radius is small, some sampling points are not covered by any mesh vertices, and this error will be caused.



## Acknowledgement
Thanks to the geodesic calculation algorithm provided by the [VTP method](https://github.com/YipengQin/VTP_source_code), we use it to calculate the local geodesics. Thanks to the code provided by [Robust non-rigid motion tracking and surface reconstruction using l0 regularization](http://www.liuyebin.com/nonrigid.html), we ran the test of SVR-L0 with minor modifications, and referenced its implementation of the node graph construction.
