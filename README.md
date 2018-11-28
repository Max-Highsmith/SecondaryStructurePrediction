#
#Use of alignment assumes the installation of PSPRO tool found here http://sysbio.rnet.missouri.edu/multicom_toolbox/tools.html


# SecondaryStructurePrediction

Deep learning secondary structure prediction

## Results

window_size=11

win_it_size = 50

inOneHotVectSize=22

outOneHotVectSize=4

### basic

dropout=0.4

learning_rate=0.0005

epochs=40

batch_size=32

loss: 0.8272695092021721

accuracy: 0.627124216780551

epochs=30

loss: 0.8168286106056293

accuracy: 0.6319711250973616

50 
loss: 0.843122710525912
accuracy: 0.621527367551243

20
loss: 0.815374844522253
accuracy: 0.6294368420992779

TODO
learning_rate=0.0008 0.0002
dropout=0.3 0.5
2 conv
batch_size=64 16
35
loss: 0.84013270550881
accuracy: 0.6195324510335922

2 conv delete 128, 3
loss: 0.8338047133659355
accuracy: 0.6178552426969431

4conv add another 128, 3
loss: 0.8439372417714412
accuracy: 0.6264666341177354

dropout 0.3
loss: 0.8578057832076118
accuracy: 0.6163516673410823

dropout 0.5
loss: 0.8159546462821402
accuracy: 0.6289122530170783

learning_rate=0.0008 
loss: 0.8355000137856475
accuracy: 0.6261489237470962

learning_rate=0.0002
loss: 0.8101210580613121
accuracy: 0.6340694825500444

batch_size 64
loss: 0.8226014097428291
accuracy: 0.6287792588551426

16
loss: 0.8395445466196584
accuracy: 0.6181507861999836

loss = 'categorical_hinge'
loss: 0.7244482176288749
accuracy: 0.6246933742136888

loss = 'mean_absolute_error'
loss: 0.2102318553954014
accuracy: 0.5795934808056593

loss = 'mean_squared_logarithmic_error'
loss: 0.060001688136177816
accuracy: 0.6287312331683304
