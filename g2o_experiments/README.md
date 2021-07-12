sudo docker build . -t pyg2o

sudo docker run -v $(pwd):/dronemapper -p 8888:8888 -it pyg2o /bin/bash


jupyter lab --allow-root --ip=0.0.0.0 --port=8888
