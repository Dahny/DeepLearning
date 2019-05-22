#!/usr/bin/env bash

# Install requirements
sudo apt install ffmpeg

# Download source file
wget http://distribution.bbb3d.renderfarming.net/video/mp4/bbb_sunflower_1080p_30fps_normal.mp4
ffmpeg -i bbb_sunflower_1080p_30fps_normal.mp4 -ss 00:01:20.0 -t 00:00:10.0 source.mp4
rm bbb_sunflower_1080p_30fps_normal.mp4
mv source.mp4 video/

for size in 750 500 250 
do
  ffmpeg -i video/source.mp4 -filter:v "crop=1080:1080:420:0,scale=$size:$size" "video/cropped_$size.mp4"
  mkdir video/frames_$size
  ffmpeg -i video/cropped_$size.mp4 -q:v 2 video/frames_$size/frame%04d.jpg  -hide_banner 
done

