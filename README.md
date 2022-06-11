# LNG站点聚类

#### 介绍
人工智能原理课-LNG站点聚类实验

·出口站点  
1.  天然气--甲烷含量超过95%  
2.  温度降低到-162℃成为液体  
3.  天然气依次通过Train，温度逐步降低  
·运输  
1.  LNG船舶  
·进口站点  
  
·LNG数据  
·数据  
1.  LNG船舶3个月，航速静止（小于1节）的数据  
2.  3590578条数据  
·数据字段  
1.  mmsi  
2.  时间：Unix时间戳（秒）  
3.  航行状态  
4.  速度  
5.  经度  
6.  纬度  
7.  吃水  
  
·目标：利用算法，快速准确的找出LNG站点  

#### 软件架构
1.  Windows 10/11 Pro
2.  Python 3.9/3.10
3.  (可选)Visual Studio Code等IDE
4.  (可选)Anaconda/Miniconda等包管理工具

#### 安装教程

1.  pip install pandas numpy matplotlib scikit-learn
2.  pip install jupyter notebook
3.  下载数据集”lng2.csv“，并移至data文件夹中（https://yunpan.360.cn/surl_y2vpVKtEjLk （提取码：ed65））  

#### 使用说明

1.  对于安装Jupyter notebook的环境，打开Run.ipynb，运行
2.  对于Windows环境，右键Run.ps1，左键点击“使用Powershell运行“
3.  对于其他环境（Linux、Macosx及其他类Unix环境），终端中输入sh ./Run.sh运行

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
