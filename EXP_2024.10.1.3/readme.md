# EXP_2024.11.1.3
## Part 1 
In this part, the target is to conduct experiment on 8-16 stage
### Problem Formulation
**A**: 8 view raw reconstruction voxel\
**B**: 16 view CB projection of **A**\
**C**: Reconstructed voxel based on **B**\
**D**: Output 16 view CB projection sequence from 8-16 DL model\
**E**: 16 view raw CB projection sequence
### Data 
B is stored in ./projections\
C is stored in ./shared_data/voxel/recons/backproj
### Experiment Procedures and Results 
<1>. Calculate the RMSE between **B/D** and **E** to evaluate the quality of **B/D**\
**Code file:** proj_compare.py\
D is closer to E compared with B

<2>. Compare **A,C** with **Raw** to illustrate the reconstruction quality\
**Code file:** voxel_compare.py\
A/C have similar RMSE to Raw

<3>. Replace the 

<4>. Mannually check the **B2/D** and compare them with **Raw**, then plot their RMSE curve to determine if there is any sample is outlier\
**Code file:** EXP_4.py\

<5>. From before experiment, we find that C2 has better quality than A
