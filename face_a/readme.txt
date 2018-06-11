***数据标签格式***
提供的训练的数据集的标签的格式为.cvs格式，每一行的第一列表示图片的文件名，后面的列表示对应的ID。如“367_0007.jgp,16 ”

***输出文件格式***
在测试时，给定一个.csv格式的文件，每一行的第一列为待识别的人脸图片的文件名。
输出的文件格式与数据集的标签相同，保存为.csv格式，每一行的第一列表示待检索(probe)的图片的文件名，后面的列表示该人脸的ID（即识别返回的ID与ground-truth中ID相同，其中不在gallery中的ID记为0）
如
"
0007.jgp, 16
0008.jgp, 0
0009.jgp, 18
"

***文件说明***

train			训练集
train.csv		训练集标签

-test_a
 |-probe 		待识别的probe图片集
 |-gallery 		gallery集
test_a_probe.csv	需要识别的probe的文件名列表
test_a_gallery		测试集gallery标签

注意：
训练集中的ID和测试集的ID相互独立