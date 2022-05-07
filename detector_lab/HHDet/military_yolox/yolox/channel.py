r,g,b = cv2.split(bgr)#拆分
rgb = cv2.merge([r,r,r])
cv2.imshow('rgb',rgb)
new_rgb = cv.imwrite('rgb.tif',rgb)