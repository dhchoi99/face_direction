# Face Landmark to Face Direction

Estimate Face direction using Face landmark  

Face landmarks are calculated with [Face Alignment](https://github.com/1adrianb/face-alignment)

Currently available methods:
```
3D Landmark(68, 3) -> Face direction
    : accurate, calculates explicit direction
2D Landmark(68, 3) -> Face direction
    : calculates implicit direction,   
      use nose angle to compare between faces 
```