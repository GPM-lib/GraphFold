chmod -R 777 ../..
docker run -it --rm --gpus all \
    --net=host \
    -v $PWD/../:/GraphhFold/GPM-artifact  -w /GraphhFold \
    graphfold:v1 /bin/bash
