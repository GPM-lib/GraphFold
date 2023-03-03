#!/bash/sh
GPU_ID=0

BIN_PATH=./build/
DATA_PATH=./datasets/



GRAPH_LIST=("cit-Patents.mtx" "livej.mtx" "youtube.mtx" "soc-pokec.mtx" "soc-lastfm.mtx" "friendster.mtx")


BIN_NAME=test_trianglecounting
OUTPUT_NAME=table3_graphfold.csv
echo "tc" |tee -a ${OUTPUT_NAME} 
for GRAPH_NAME in ${GRAPH_LIST[@]}
do
	echo $GRAPH_NAME |tee -a ${OUTPUT_NAME} 
	CUDA_VISIBLE_DEVICES=$GPU_ID ${BIN_PATH}${BIN_NAME} ${DATA_PATH}${GRAPH_NAME} | grep "counting  time" |grep -oP '\d*\.\d+' |tee -a ${OUTPUT_NAME} 
done

BIN_NAME=test_motifcounting
OUTPUT_NAME=table4_graphfold.csv
echo "4-motif" |tee -a ${OUTPUT_NAME} 
for GRAPH_NAME in ${GRAPH_LIST[@]}
do
	echo $GRAPH_NAME |tee -a ${OUTPUT_NAME} 
	CUDA_VISIBLE_DEVICES=$GPU_ID ${BIN_PATH}${BIN_NAME} ${DATA_PATH}${GRAPH_NAME}  4 | grep "counting  time" |grep -oP '\d*\.\d+' |tee -a ${OUTPUT_NAME} 
done


BIN_NAME=test_cliquefinding
OUTPUT_NAME=table5_graphfold.csv
echo "4-clique" |tee -a ${OUTPUT_NAME} 
for GRAPH_NAME in ${GRAPH_LIST[@]}
do
	echo $GRAPH_NAME |tee -a ${OUTPUT_NAME} 
	CUDA_VISIBLE_DEVICES=$GPU_ID ${BIN_PATH}${BIN_NAME} ${DATA_PATH}${GRAPH_NAME} 4  $ | grep "matching  time" |grep -oP '\d*\.\d+' |tee -a ${OUTPUT_NAME} 
done

BIN_NAME=test_cliquefinding
OUTPUT_NAME=table5_graphfold.csv
echo "5-clique" |tee -a ${OUTPUT_NAME} 
for GRAPH_NAME in ${GRAPH_LIST[@]}
do
	echo $GRAPH_NAME |tee -a ${OUTPUT_NAME} 
	CUDA_VISIBLE_DEVICES=$GPU_ID ${BIN_PATH}${BIN_NAME} ${DATA_PATH}${GRAPH_NAME} 5  $ | grep "matching  time" |grep -oP '\d*\.\d+' |tee -a ${OUTPUT_NAME} 
done

BIN_NAME=test_cliquefinding
OUTPUT_NAME=table5_graphfold.csv
echo "6-clique" |tee -a ${OUTPUT_NAME} 
for GRAPH_NAME in ${GRAPH_LIST[@]}
do
	echo $GRAPH_NAME |tee -a ${OUTPUT_NAME} 
	CUDA_VISIBLE_DEVICES=$GPU_ID ${BIN_PATH}${BIN_NAME} ${DATA_PATH}${GRAPH_NAME} 6  $ | grep "matching  time" |grep -oP '\d*\.\d+' |tee -a ${OUTPUT_NAME} 
done

BIN_NAME=test_cliquefinding
OUTPUT_NAME=table5_graphfold.csv
echo "7-clique" |tee -a ${OUTPUT_NAME} 
for GRAPH_NAME in ${GRAPH_LIST[@]}
do
	echo $GRAPH_NAME |tee -a ${OUTPUT_NAME} 
	CUDA_VISIBLE_DEVICES=$GPU_ID ${BIN_PATH}${BIN_NAME} ${DATA_PATH}${GRAPH_NAME} 7  $ | grep "matching  time" |grep -oP '\d*\.\d+' |tee -a ${OUTPUT_NAME} 
done