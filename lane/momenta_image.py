import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

#warnings.simplefilter('ignore', np.RankWarning)

name = '77b10b70ee6a839bed3ae0da98727578'

json_url = os.path.join('guangqi_results_lane', name+'.json')
image_url = os.path.join('images',name+'.jpg')

with open(json_url,'r') as jsonfile:
    str_json = json.load(jsonfile)

image = cv2.imread(image_url)
height, width, channel = image.shape

# len(str_json['Lines']) = line number that detected
# len(str_json['lines'][0]['cpoints']) = first line point number
# str_json['lines'][0]['cpoints'][0]: first line's first number

lines = []
lines_x = []
lines_y = []

for i in str_json['Lines']:
    x_point = []
    y_point = []
    for j in i['cpoints']:
        #print(j)
        x_point.append(int(j['x']))
        y_point.append(int(j['y']))
    x = np.array(x_point)
    y = np.array(y_point)
    lines_x.append(x)
    lines_y.append(y)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        n = 1
        try:
            z = np.polyfit(y,x,1)
            n = 1
        except np.RankWarning:
            try:
                z = np.polyfit(y,x,2)
                n = 2
            except np.RankWarning:
                warnings.simplefilter('ignore',np.RankWarning)
                z = np.polyfit(y,x,3)
                n = 3
        if n==1:
            lines.append([0, 0, z[0], z[1]])
        elif n==2:
            lines.append([0, z[0], z[1], z[2]])
        else:
            lines.append(z)
j = 0
ploty = np.linspace(0, height-1, 72)
plt.figure()
for i in range(len(str_json['Lines'])):
    curv = lines[j][0]*ploty**3 + lines[j][1]*ploty**2 + lines[j][2]*ploty+lines[j][3]
    x = []
    y = []
    for k in range(len(curv)):
        if curv[k]>=0 and curv[k]<width and ploty[k]>min(lines_y[i]):
            x.append(curv[k])
            y.append(ploty[k])
            cv2.circle(image, (int(curv[k]), int(ploty[k])), 5, (255,0,255), 3)
    plt.plot(x, y)
    j=j+1
plt.show()
cv2.imwrite('save.jpg', image)




