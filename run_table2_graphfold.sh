#!/bash/sh
GPU_ID=0

BIN_PATH=./build/
DATA_PATH=./datasets/

BIN_NAME=test_patternEnum
OUTPUT_NAME=table2_graphfold.csv
PATTERN_LIST=("P1"  "P2"  "P3"  "P4"  "P5"  "P6"  "P7"  "P8")

GRAPH_LIST=("cit-Patents.mtx"  "livej.mtx" "youtube.mtx" "mico.mtx" "soc-LiveMocha.mtx" "soc-gowalla.mtx" "enron.mtx" "soc-delicious.mtx" "soc-pokec.mtx" "orkut.mtx" "soc-lastfm.mtx")
#GRAPH_LIST=("cit-Patents.mtx"   "youtube.mtx" "mico.mtx")
for GRAPH_NAME in ${GRAPH_LIST[@]}
do
	echo $GRAPH_NAME |tee -a ${OUTPUT_NAME} 
	for PATTERN_NAME in ${PATTERN_LIST[@]}
	do
		CUDA_VISIBLE_DEVICES=$GPU_ID ${BIN_PATH}${BIN_NAME} ${DATA_PATH}${GRAPH_NAME} $PATTERN_NAME | grep "matching  time" |grep -oP '\d*\.\d+' |tee -a ${OUTPUT_NAME} 
	done
done
