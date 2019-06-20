[X1,map1]=imread('LATENCY_T=GEANT@A=0.6.jpg');
[X2,map2]=imread('LATENCY_T=GEANT@A=0.8.jpg');
[X3,map3]=imread('LATENCY_T=GEANT@A=1.0.jpg');
[X4,map4]=imread('LATENCY_T=GEANT@A=1.2.jpg');


subplot(2,2,1), imshow(X1,map1)
subplot(2,2,2), imshow(X2,map2)
subplot(2,2,3), imshow(X3,map1)
subplot(2,2,4), imshow(X4,map2)