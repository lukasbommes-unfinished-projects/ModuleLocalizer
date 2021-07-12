sudo docker build ./Docker -t pyg2o

sudo docker run -it \
    --ipc=host \
    --env="DISPLAY" \
    --gpus=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd):/dronemapper \
    -v /storage/:/storage \
    -p "8888:8888" \
    pyg2o \
    bash

jupyter lab --allow-root --ip=0.0.0.0 --port=8888
