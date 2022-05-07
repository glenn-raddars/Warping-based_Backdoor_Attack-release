train.py                          训练WaNet
adversarial_finetune.py   AFT防御WaNet
train_mutli-blend.py       训练Blend后门
aft_mutli-blend.py          AFT防御Blend
see_percentage.py          计算all2all混淆矩阵
see_percentage_all2one.py    计算all2one混淆矩阵


如何更改Blend攻击设置
train_mutli-blend.py
line 38
更改create_bd函数即可