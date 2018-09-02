import numpy as np
import json
import time


# 目标：选择在当前局面下能够最大程度提高赢的概率的出牌
#
# 过程：
#   1. 还原当前局面
#   2. （以某些顺序）找出可能的出牌方式
#   3. 对每种方式使用训练好的模型计算赢的概率
#   4. 选择赢的概率最大的出牌方式


# 对局json输出示例：
# {"response":[24,25]}


# 为了计算参数必需的信息：
#   我是谁：myPosition
#   三家现在手里有多少牌：cardRemaining
#   我手里有什么牌：myCard
#   我需要管什么牌：lastValid
#   现在是第几轮：turn
# 输入：我准备出什么牌
# 输出：出完牌后赢的概率
# 计算中输入的参数应该是模拟出完牌之后的！


# 准备数据
full_input = json.loads(input())
t0 = time.monotonic()
myCard: list = full_input["requests"][0]["own"]
myCard.sort()
first_history = full_input["requests"][0]["history"]
lastValid = []
if len(first_history[0]) == 0:      # 判断我是谁
    if len(first_history[1]) == 0:
        myPosition = 0
    else:
        myPosition = 1
else:
    myPosition = 2
turn = len(full_input["requests"])
cardRemaining = [20, 17, 17]
for i in range(0, turn):            # 还原其他两个人手里现在有的牌
    history = full_input["requests"][i]["history"]
    howManyPass = 0
    for p in [0, 1]:
        player = (myPosition + 1 + p) % 3
        cardRemaining[player] -= len(history[p])
        if len(history[p]) == 0:
            howManyPass += 1
        else:
            lastValid = history[p]
        if howManyPass == 2:
            lastValid = []
for action in full_input["responses"]:  # 还原我手里现在有的牌
    for card in action:
        myCard.remove(card)
    cardRemaining[myPosition] -= len(action)


# 导入参数
input_layer_size = 540
hidden_layer_size = 30
theta1 = np.empty((hidden_layer_size, input_layer_size + 1), dtype=np.double)
with open("theta1.txt", "r") as f:
    for i in range(0, theta1.shape[0]):
        line = f.readline().split(" ")
        for j in range(0, len(line) - 1):
            theta1[i][j] = float(line[j])
theta2 = np.empty((hidden_layer_size, hidden_layer_size + 1), dtype=np.double)
with open("theta2.txt", "r") as f:
    for i in range(0, theta2.shape[0]):
        line = f.readline().split(" ")
        for j in range(0, len(line) - 1):
            theta2[i][j] = float(line[j])
theta3 = np.empty((1, hidden_layer_size + 1), dtype=np.double)
with open("theta3.txt", "r") as f:
    for i in range(0, theta3.shape[0]):
        theta3[i] = float(f.readline())


# 分牌器
def findAllValid():
    pass


# 计算输入参数
def calculatePara():
    global para1, para6, para7
    return np.array([0])


# 计算sigmoid函数值
def sigmoid(z: np.ndarray):
    return 1 / (1 + np.exp(-z))


# 计算概率
def calculateProb(para, theta1, theta2, theta3):
    h1 = sigmoid(np.hstack((np.array([1]), para)) * theta1.transpose())
    h2 = sigmoid(np.hstack((np.array([1]), h1)) * theta2.transpose())
    h3 = sigmoid(np.hstack((np.array([1]), h2)) * theta3.transpose())
    return h3


def card2level(card):
    return card // 4 + card // 53


def cardAnalysis(cards):
    levels = [0] * 15
    para4 = [0] * 54
    for each in cards:
        levels[card2level(each)] += 1
    for i in range(0, 14):
        if levels[i] != 0:
            para4[i * 4 + levels[i]] = 1
    if levels[13] == 1:
        para4[52] = 1
    if levels[14] == 1:
        para4[53] = 1
    return para4


def findMaxSeq(packs):
    for c in range(1, len(packs)):
        if packs[c].count != packs[0].count or packs[c].level != packs[c - 1].level - 1:
            return c
    return len(packs)


class cardPack:
    def __init__(self, level, count):
        self.level = level
        self.count = count

    def __lt__(self, other):
        if self.count == other.count:
            return self.level > other.level
        return self.count > other.count


class cardCombo:
    def __init__(self, cards):
        self.combotype = ""
        self.combolevel = 0
        self.cards = cards
        self.packs = []
        if len(cards) == 0:
            self.combotype = "pass"
            return
        counts = [0] * 15
        countOfCount = [0] * 5
        for each in cards:
            counts[card2level(each)] += 1
        for l in range(0, 15):
            if counts[l] != 0:
                self.packs.append(cardPack(l, counts[l]))
                countOfCount[counts[l]] += 1
        self.packs.sort()
        self.combolevel = self.packs[0].level
        kindOfCountOfCount = []
        for i in range(0, 5):
            if countOfCount[i] != 0:
                kindOfCountOfCount.append(i)
        kindOfCountOfCount.sort()

        if len(kindOfCountOfCount) == 1:  # 只有一类牌
            curr = countOfCount[kindOfCountOfCount[0]]
            if kindOfCountOfCount[0] == 1:  # 只有若干单张
                if curr == 1:
                    self.combotype = "single"
                elif curr == 2 and self.packs[1].level == 13:
                    self.combotype = "rocket"
                elif 5 <= curr == findMaxSeq(self.packs) and self.packs[0].level <= 11:
                    self.combotype = "straight"
            elif kindOfCountOfCount[0] == 2:  # 只有若干对子
                if curr == 1:
                    self.combotype = "pair"
                if 3 <= curr == findMaxSeq(self.packs) and self.packs[0].level <= 11:
                    self.combotype = "straight2"
            elif kindOfCountOfCount[0] == 3:  # 只有若干三条
                if curr == 1:
                    self.combotype = "triplet"
                elif curr == findMaxSeq(self.packs) and self.packs[0].level <= 11:
                    self.combotype = "plane"
            elif kindOfCountOfCount[0] == 4:  # 只有若干四条
                if curr == 1:
                    self.combotype = "bomb"
                elif curr == findMaxSeq(self.packs) and self.packs[0].level <= 11:
                    self.combotype = "sshuttle"
        elif len(kindOfCountOfCount) == 2:  # 有两类牌
            curr = countOfCount[kindOfCountOfCount[1]]
            lesser = countOfCount[kindOfCountOfCount[0]]
            if kindOfCountOfCount[1] == 3:  # 三条带？
                if curr == 1 and lesser == 1:
                    self.combotype = "triplet1"
                elif findMaxSeq(self.packs) == curr == lesser and self.packs[0].level <= 11:
                    self.combotype = "plane1"
            if kindOfCountOfCount[1] == 4:  # 四条带？
                if kindOfCountOfCount[0] == 1:
                    if curr == 1 and lesser == 2:
                        self.combotype = "quardruple2"
                    if findMaxSeq(self.packs) == curr and lesser == curr * 2 and self.packs[0].level <= 11:
                        self.combotype = "sshuttle2"
                if kindOfCountOfCount[0] == 2:
                    if curr == 1 and lesser == 2:
                        self.combotype = "quardruple4"
                    if findMaxSeq(self.packs) == curr and lesser == curr * 2 and self.packs[0].level <= 11:
                        self.combotype = "sshuttle4"
        if self.combotype == "":
            raise ValueError


def myAction(cards):
    para5 = [0] * 232
    myaction = cardCombo(cards)

    #acc = 0

    # 过
    if myaction.combotype == "pass":
        para5[0] = 1
        #acc += 1

    # 单张
    # 使用level的一次、两次和三次项
    if myaction.combotype == "single":
        para5[1+myaction.combolevel] = 1
        #acc += 15

    # 对子
    # 使用level的一次、两次和三次项
    if myaction.combotype == "pair":
        para5[16+myaction.combolevel] = 1
        #acc += 15

    # 顺子
    # 顺子起点的level有[0-7]共8种，终点有[4-11]共8种，共64种顺子
    if myaction.combotype == "straight":
        end = myaction.combolevel + findMaxSeq(myaction.packs) - 1
        para5[31+myaction.combolevel*8+end-4] = 1
        #acc += 64

    # 双顺
    # 双顺和单顺作同样处理
    if myaction.combotype == "straight2":
        end = myaction.combolevel + findMaxSeq(myaction.packs) - 1
        para5[95+myaction.combolevel*8+end-4] = 1
        #acc += 64

    # 三条
    # 因为纯三条比较少见，所以采用一次项和二次项
    if myaction.combotype == "triplet":
        para5[159] = myaction.combolevel
        para5[160] = myaction.combolevel
        #acc += 2

    # 三带一
    # 采用三条和单张的一次项
    if myaction.combotype == "triplet1":
        para5[161+myaction.packs[0].level] = 1
        #acc += 15
        para5[176+myaction.packs[1].level] = 1
        #acc += 15

    # 三带二
    # 与三带一类似
    if myaction.combotype == "triplet2":
        para5[191+myaction.packs[0].level] = 1
        #acc += 15
        para5[206+myaction.packs[1].level] = 1
        #acc += 15

    # 炸弹
    # 采用炸弹的一次项和二次项
    if myaction.combotype == "bomb":
        para5[221] = myaction.combolevel
        para5[222] = myaction.combolevel ** 2
        #acc += 2

    # 四带二（只）
    # 以下都比较稀少，只采用一个有或没有
    if myaction.combotype == "quadruple2":
        para5[223] = 1
        #acc += 1

    # 四带二（对）
    if myaction.combotype == "quadruple4":
        para5[224] = 1
        #acc += 1

    # 飞机
    if myaction.combotype == "plane":
        para5[225] = 1
        #acc += 1

    # 飞机带小翼
    if myaction.combotype == "plane1":
        para5[226] = 1
        #acc += 1

    # 飞机带大翼
    if myaction.combotype == "plane2":
        para5[227] = 1
        #acc += 1

    # 航天飞机
    if myaction.combotype == "sshuttle":
        para5[228] = 1
        #acc += 1

    # 航天飞机带小翼
    if myaction.combotype == "sshuttle2":
        para5[229] = 1
        #acc += 1

    # 航天飞机带大翼
    if myaction.combotype == "sshuttle4":
        para5[230] = 1
        #acc += 1

    # 火箭
    if myaction.combotype == "rocket":
        para5[231] = 1
        #acc += 1

    return para5


# 计算通用的参数
para1 = [0] * 3
para1[myPosition] = 1
para6 = myAction(lastValid)
para7 = [turn]


# 迭代计算最优解
pmin = 0
myOutput = []
for each in findAllValid():
    para = calculatePara(each)
    p_curr = calculateProb(para, theta1, theta2, theta3)
    if p_curr > pmin:
        myAction = each

print(json.dumps({"response": myOutput}))


t1 = time.monotonic()
print(t1 - t0)

