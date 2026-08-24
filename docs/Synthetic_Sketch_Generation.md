**Synthetic Sketch Generation.**<br>
Because collecting real sketches is expensive and tedious, we propose a fully automatic pipeline to generate synthetic VR sketches from 3D meshes, producing 20,838 samples in roughly 10 hours on a standard workstation.

- Extracting Salient Points.<br> 
We begin by uniformly sampling 2048 points on the surface of the input mesh. Sketches typically emphasize visually prominent geometric features such as edges, corners, and holes. We emphasize regions of high curvature and structural significance by extracting the salient point cloud using Sharp Edge Sampling (SES) and a curvature threshold of 15.<br>
<a href="https://github.com/Seed3D/Dora">Sharp Edge Sampling (SES) code</a><br>

- Recovering Strokes.<br> 
We then fit Bézier splines to the salient point cloud using EMAP with a maximum degree of 2 and minimum segment length of 12. The points along each spline form the individual strokes. Next, we apply a culling stage inspired by the approach proposed by Liu et al. We first remove redundant points in near-linear segments with a cosine distance threshold of 0.04. Finally, we merge strokes whose endpoints lie within a threshold of 2% of the normalized shape size.<br>
<a href="https://github.com/cvg/EMAP">EMAP code</a><br>
<a href="https://github.com/davepagurek/StrokeStrip">Culling stage inspiration</a><br>

- Ordering Strokes.<br> 
To approximate human drawing order, we connect stroke endpoints based on spatial proximity and perform a depth-first traversal of the resulting connectivity graph. We introduce stochasticity by skipping nearest connections with a probability of 10%, yielding coherent yet varied stroke sequences.