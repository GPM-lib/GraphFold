#!/bash/sh
GPU_ID=0
BIN_PATH=./build/
DATA_PATH=./datasets/



GRAPH_LIST=("cit-Patents.mtx" "livej.mtx" "youtube.mtx" "soc-pokec.mtx" "soc-lastfm.mtx")

BIN_NAME=test_G2Miner
OUTPUT_NAME=table3_g2miner.csv
echo "tc" |tee -a ${OUTPUT_NAME} 
for GRAPH_NAME in ${GRAPH_LIST[@]}
do
	echo $GRAPH_NAME |tee -a ${OUTPUT_NAME} 
	CUDA_VISIBLE_DEVICES=$GPU_ID ${BIN_PATH}${BIN_NAME} ${DATA_PATH}${GRAPH_NAME} TC | grep "matching  time" |grep -oP '\d*\.\d+' |tee -a ${OUTPUT_NAME} 
done

BIN_NAME=test_G2Miner
OUTPUT_NAME=table4_g2miner.csv
echo "4-motif" |tee -a ${OUTPUT_NAME} 
for GRAPH_NAME in ${GRAPH_LIST[@]}
do
	echo $GRAPH_NAME |tee -a ${OUTPUT_NAME} 
	CUDA_VISIBLE_DEVICES=$GPU_ID ${BIN_PATH}${BIN_NAME} ${DATA_PATH}${GRAPH_NAME}  MC4 | grep "matching  time" |grep -oP '\d*\.\d+' |tee -a ${OUTPUT_NAME} 
done


BIN_NAME=test_G2Miner
OUTPUT_NAME=table5_g2miner.csv
echo "4-clique" |tee -a ${OUTPUT_NAME} 
for GRAPH_NAME in ${GRAPH_LIST[@]}
do
	echo $GRAPH_NAME |tee -a ${OUTPUT_NAME} 
	CUDA_VISIBLE_DEVICES=$GPU_ID ${BIN_PATH}${BIN_NAME} ${DATA_PATH}${GRAPH_NAME} CF4 $ | grep "matching  time" |grep -oP '\d*\.\d+' |tee -a ${OUTPUT_NAME} 
done

BIN_NAME=test_G2Miner
OUTPUT_NAME=table5_g2miner.csv
echo "5-clique" |tee -a ${OUTPUT_NAME} 
for GRAPH_NAME in ${GRAPH_LIST[@]}
do
	echo $GRAPH_NAME |tee -a ${OUTPUT_NAME} 
	CUDA_VISIBLE_DEVICES=$GPU_ID ${BIN_PATH}${BIN_NAME} ${DATA_PATH}${GRAPH_NAME} CF5 $ | grep "matching  time" |grep -oP '\d*\.\d+' |tee -a ${OUTPUT_NAME} 
done

BIN_NAME=test_G2Miner
OUTPUT_NAME=table5_g2miner.csv
echo "6-clique" |tee -a ${OUTPUT_NAME} 
for GRAPH_NAME in ${GRAPH_LIST[@]}
do
	echo $GRAPH_NAME |tee -a ${OUTPUT_NAME} 
	CUDA_VISIBLE_DEVICES=$GPU_ID ${BIN_PATH}${BIN_NAME} ${DATA_PATH}${GRAPH_NAME} CF6  $ | grep "matching  time" |grep -oP '\d*\.\d+' |tee -a ${OUTPUT_NAME} 
done

BIN_NAME=test_G2Miner
OUTPUT_NAME=table5_g2miner.csv
echo "7-clique" |tee -a ${OUTPUT_NAME} 
for GRAPH_NAME in ${GRAPH_LIST[@]}
do
	echo $GRAPH_NAME |tee -a ${OUTPUT_NAME} 
	CUDA_VISIBLE_DEVICES=$GPU_ID ${BIN_PATH}${BIN_NAME} ${DATA_PATH}${GRAPH_NAME} CF7  $ | grep "matching  time" |grep -oP '\d*\.\d+' |tee -a ${OUTPUT_NAME} 
done