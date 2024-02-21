docker build . -t dev-vall-e
docker run --gpus all -v /mnt:/mnt -i -t --name dev-valle-container dev-vall-e