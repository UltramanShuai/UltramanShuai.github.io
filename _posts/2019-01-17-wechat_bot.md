---
layout:     post
title:      Wechat bot
subtitle:   Autorun robot compresses images by Mini Batch K-Means
date:       2019-01-15
author:     ShuaiGao
header-img: img/post-bg-ios9-web.jpg
catalog: False
tags:
    - Mini Batch K-Means
    - Machine Learning
    - image compression
---

# Autorun robot compresses images by Mini Batch K-Means
[Look at the code here](https://github.com/UltramanShuai/wechat_robot)

Using `itchat` package connect to wechat and recieve picture from friends, by using Mini Batch K-Means alorithm to replace the colors that picture contained with the color in the same cluster. The number of colors left can be easily defined. With a suitable color left number, the converted picture size has been significantly reduced without distortion. 

![IMG_0488.jpg](https://i.loli.net/2019/01/15/5c3dfbf53df0c.jpg)
The right hand side only contains 5 colors. 

Except reduce the size of picture, it can also be used to produce something interesting.
![IMG_0487.jpg](https://i.loli.net/2019/01/15/5c3dfbb3a31a3.jpg)


