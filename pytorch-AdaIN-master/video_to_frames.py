import cv2
vidcap = cv2.VideoCapture('input/videos/cutBunny.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("input/frames/frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1

print("There are a total of:", count, "frames!!!")