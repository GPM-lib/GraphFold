#!/bash/sh
GPU_ID=0
BIN_PATH=./build/
DATA_PATH=./datasets/

BIN_NAME=test_patternEnum_multigpu
OUTPUT_NAME=figure7_graphfold.csv

PATTERN_LIST=("P4" "P7" "P8")
GRAPH_LIST=("livej.mtx"   "orkut.mtx" "soc-pokec.mtx")
GPU_LIST=("1" "2" "3" "4" "5" "6" "7" "8")

for GRAPH_NAME in ${GRAPH_LIST[@]}
do
	echo $GRAPH_NAME |tee -a ${OUTPUT_NAME} 
	for PATTERN_NAME in ${PATTERN_LIST[@]}
	do
		echo $PATTERN_NAME |tee -a ${OUTPUT_NAME} 
		for GPU_NUM in ${GPU_LIST[@]}
		do
			echo "${BIN_PATH}${BIN_NAME} ${DATA_PATH}${GRAPH_NAME} ${PATTERN_NAME} ${GPU_NUM}"
			${BIN_PATH}${BIN_NAME} ${DATA_PATH}${GRAPH_NAME} ${PATTERN_NAME} ${GPU_NUM} | grep "GPU0" |grep -oP '\d*\.\d+' |tee -a ${OUTPUT_NAME} 
		done
	done
done
