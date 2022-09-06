# SSUNet
This paper is being submitted for UIC2022 named "Shape Strengthed U-shape Network for Objects Extraction of Remote Sensing Images". Any questions could be answered if we can. The E-mail is 1091007069@qq.com.

For the network, it has two streams (body stream and shape stream). In body stream, the U-Net with Channel attention help the model obtain the body information. In shape stream, the lost shape information should be recovered.

![图片1](https://user-images.githubusercontent.com/80099298/188646595-e7ec80be-d908-48e9-be6b-4320287327c4.png)


In this work, you should have a GPU whose video memory is over 8 GB. The input images and labels should be 512×512 (pixels * pixels). If not, you should change the relative parameters in Vit and GBE. (best: 3090 or 2080Ti)

For w1 (the weight of boundary loss), we try 1,5,10, and 25 where 5 brought the best performance.



### step1:
make images and labels to the suitable size.

### step2:
`python train_two.py` (two branch)
and you can change the batchsize, lr, name, dir, and epoch in config.txt

### Another  step:
`python train_one.py`  (one branch)


## To be Updated...
