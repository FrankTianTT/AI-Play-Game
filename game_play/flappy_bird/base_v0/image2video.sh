ffmpeg -i image/%5d.jpg -vcodec libx264 -r 30  image.mp4
ffmpeg -i nn_graph/relu1/%5d.jpg -vcodec libx264 -r 30  relu1.mp4
ffmpeg -i nn_graph/relu2/%5d.jpg -vcodec libx264 -r 30  relu2.mp4
ffmpeg -i nn_graph/relu3/%5d.jpg -vcodec libx264 -r 30  relu3.mp4
ffmpeg -i nn_graph/relu4/%5d.jpg -vcodec libx264 -r 30  relu4.mp4
ffmpeg -i nn_graph/relu5/%5d.jpg -vcodec libx264 -r 30  relu5.mp4