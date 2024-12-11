# Zero-Shot Self-Consistancy Learning for Seismic Reconstrutiong
## 1. Structure of Code

```
filetree 
├── README.md
├── data
│  ├── /bs_mat/  % Research lines in matlab format from the USGS Beaufort Sea-Artic Alaska project
│  ├── /ca_mat/  % Research lines in matlab format from the USGS National Petroleum Reserve–Alaska
│  └── /examples/  % Examples for visualization in paper
├── ./func/  % Main functions  
├── main.py    % Demo to reconstruct  
├── visualization_Line22.py   % The visualization of example Line 22
├── visualization_Line33.py   % The visualization of example Line 33
├── visualization_WB905m.py   % The visualization of example Line WB905m
└── visualization_WB905m.py   % The visualization of example Line WB905m
```

## 2. Test Environment
### Hardware
```
Intel Core i5 14400F
16GB memory
NVIDIA RTX4060 with 6GB memory
```
### Software
```
cudatoolkit               11.8.0 
cudnn                     8.9.2.26
python                    3.8.20 
pytorch                   2.4.1 
scipy                     1.10.1 
```