# Artifact for PPoPP'24 paper 
> *Exploiting Fine-Grained Redundancy in Set-Centric Graph Pattern Mining.* 


## 1.1. Download datasets.

+ Download datasets.
Datasets are available [here](https://drive.google.com/drive/folders/10zlTSeErz9_egkeuU_NhpUnNNwzoh8U3?usp=share_link). 

+ Download single dataset(cit-Patents.mtx).
```
cd datasets
bash fetch_cit-Patents.sh
```

+ Batch download datasets.
```
cd datasets
bash fetch_all_data.sh
```

## 1.2. Compile implementation.
```
mkdir build && cd build && cmake .. && make -j10 && cd ..
```

# 2. Run initial test experiment with basic APIs.
## 2.1. Run baseline G<sup>2</sup>Miner.
>Usage: ./build/test_G2Miner <graph_path> <pattern_name>  
>Support Patterns(pattern_name): K-Clique-Finding(CF4,CF5,CF6,CF7), Motif-Counting(MC4), Triangle-Counting(TC), Pattern-Enumeration(P1,P2,P3,P4,P5,P6,P7,P8)

```
./build/test_G2Miner ./datasets/youtube.mtx CF4
./build/test_G2Miner ./datasets/cit-Patents.mtx MC4
./build/test_G2Miner ./datasets/cit-Patents.mtx TC
./build/test_G2Miner ./datasets/cit-Patents.mtx P1
```

## 2.2. Run GraphFold's Pattern-Enumeration.
>Usage: ./build/test_patternEnum <graph_path> <pattern_name>  
>Support Patterns(pattern_name): Pattern-Enumeration(P1,P2,P3,P4,P5,P6,P7,P8)

```
./build/test_patternEnum  ./datasets/cit-Patents.mtx P1
```

## 2.3. Run GraphFold's K-Clique-Finding.
>Usage: ./build/test_cliquefinding <graph_path> <n>  
>Support Patterns(pattern_name): K-Clique-Finding(4,5,6,7)

```
./build/test_cliquefinding  ./datasets/cit-Patents.mtx 5
```

## 2.4. Run GraphFold's Motif-Counting.
>Usage: Usage: ./build/test_motifcounting <graph_path> <n>

>Support Patterns(pattern_name): Motif-Counting(4)

```
./build/test_motifcounting  ./datasets/cit-Patents.mtx 4
```

## 2.5. Run GraphFold's Triangle-Counting.
>Usage: ./build/test_trianglecounting <graph_path> 

>Support Patterns(pattern_name): Triangle-Counting()

```
./build/test_trianglecounting  ./datasets/cit-Patents.mtx
```

## 2.6 Run GraphFold on multi-GPUs.
>Usage: ./build/test_$BIN_NAME$_multigpu <graph_path> <pattern_name> <num_gpus>  
>Support all patterns on multi-GPUs, and support number of GPUs(1,2,3,4,5,6,7,8).
```
./build/test_patternEnum_multigpu ./datasets/cit-Patents.mtx P5 4
```

# 3. Reproduce the major results from paper.
## 3.1 Compare with G<sup>2</sup>Miner on V100 GPU. (Table 2.The runtime of GraphFold vs. G<sup>2</sup>Miner) 
```
bash run_table2_graphfold.sh
```
> Note that the results of GraphFold can be found at "table2_graphfold.csv". Set DATA_PATH(where the graph datasets dir), GRAPH_LIST(which graph runs), GPU_ID(which GPU use), OUTPUT_NAME(where results output), PATTERN_LIST(which pattern runs).

```
bash run_table2_g2miner.sh
```
> Note that the results of G<sup>2</sup>Miner can be found at "table2_g2miner.csv". Set DATA_PATH(where the graph datasets dir), GRAPH_LIST(which graph runs), GPU_ID(which GPU use), OUTPUT_NAME(where results output), PATTERN_LIST(which pattern runs).

## 3.2 Compare with G<sup>2</sup>Miner on V100 GPU. (Table 3.4.5) 
```
bash run_multi_tables_graphfold.sh
```
> Note that the results of GraphFold can be found at "tablexx(3,4,5)_graphfold.csv". Set DATA_PATH(where the graph datasets dir),  OUTPUT_NAME(where results output).

```
bash run_multi_tables_g2miner.sh
```
> Note that the results of G<sup>2</sup>Miner can be found at "tablexx(3,4,5)_g2miner.csv". Set DATA_PATH(where the graph datasets dir),  OUTPUT_NAME(where results output).

## 3.3 Compare with G<sup>2</sup>Miner on 8*V100 GPU. (Figure7.) 
```
bash run_fig7_graphfold.sh
```
> Note that the results of GraphFold can be found at "figure7_graphfold.csv". Set DATA_PATH(where the graph datasets dir), GRAPH_LIST(which graph runs), GPU_LIST(number of GPUs), OUTPUT_NAME(where results output), PATTERN_LIST(which pattern runs).

```
bash run_fig7_g2miner.sh
```
> Note that the results of G<sup>2</sup>Miner can be found at "figure7_g2miner.csv". Set DATA_PATH(where the graph datasets dir), GRAPH_LIST(which graph runs), GPU_LIST(number of GPUs), OUTPUT_NAME(where results output), PATTERN_LIST(which pattern runs).

# 4. Run GraphFold in docker.
## 4.1 Launch the GraphFold docker
```
cd docker 
./build.sh
```

## 4.2 Launch the GraphFold docker and recompile, 
+ The compiled exectuable will be located under `GPM-artifact/`.
```
cd docker 
./launch.sh
cd GPM-artifact && mkdir build && cd build && cmake .. && make -j10
```
