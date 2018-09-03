import numpy as np
import json
import time
import copy

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
if len(first_history[0]) == 0:  # 判断我是谁
    if len(first_history[1]) == 0:
        myPosition = 0
    else:
        myPosition = 1
else:
    myPosition = 2
turn = len(full_input["requests"])
cardRemaining = [20, 17, 17]
for i in range(0, turn):  # 还原其他两个人手里现在有的牌
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
with open("data/theta1.txt", "r") as f:
    for i in range(0, theta1.shape[0]):
        line = f.readline().split(" ")
        for j in range(0, len(line) - 1):
            theta1[i][j] = float(line[j])
theta2 = np.empty((hidden_layer_size, hidden_layer_size + 1), dtype=np.double)
with open("data/theta2.txt", "r") as f:
    for i in range(0, theta2.shape[0]):
        line = f.readline().split(" ")
        for j in range(0, len(line) - 1):
            theta2[i][j] = float(line[j])
theta3 = np.empty((1, hidden_layer_size + 1), dtype=np.double)
with open("data/theta3.txt", "r") as f:
    for i in range(0, theta3.shape[0]):
        theta3[i] = float(f.readline())


def canBeat(combo1, combo2):
    combo1 = cardCombo(combo1)
    if combo1.combotype == "invalid" or combo2.combotype == "invalid":
        return False
    if combo2.combotype == "rocket":
        return False
    if combo2.combotype == "bomb":
        if combo1.combotype == "rocket":
            return True
        elif combo1.combotype == "bomb":
            return combo1.combolevel > combo2.combolevel
        else:
            return False
    return combo1.combotype == combo2.combotype \
           and len(combo1.cards) == len(combo2.cards) \
           and combo1.combolevel > combo2.combolevel


def card2level(card):
    return card // 4 + card // 53


counts = [0] * 15
Single, Pair, Triplet, Bomb = [], [], [], []
for each in myCard:
    counts[card2level(each)] += 1
for i in range(0, 15):
    if counts[i] == 1:
        Single.append(i)
    elif counts[i] == 2:
        Pair.append(i)
    elif counts[i] == 3:
        Triplet.append(i)
    elif counts[i] == 4:
        Bomb.append(i)


def findSingle(myCard):
    global counts
    temp = copy.copy(counts)
    for card in myCard:
        if temp[card2level(card)] != 0:
            yield [card]
            temp[card2level(card)] = 0


def findPair(myCard):
    global counts, Pair, Triplet, Bomb
    for pair in Pair:
        result = []
        for card in myCard:
            if card2level(card) == pair:
                result.append(card)
        yield result
    for pair in Triplet:
        result = []
        for card in myCard:
            if card2level(card) == pair:
                result.append(card)
                if len(result) == 2:
                    yield result
                    continue
    for pair in Bomb:
        result = []
        for card in myCard:
            if card2level(card) == pair:
                result.append(card)
                if len(result) == 2:
                    yield result
                    continue


def findStraight(myCard):
    global counts
    for begin in range(0, 7):
        for end in range(begin + 5, 12):
            flag = True
            result = []
            for iter in range(begin, end + 1):
                if counts[iter] == 0:
                    flag = False
                    break
            if flag:
                levels = [iter for iter in range(begin, end + 1)]
                for card in myCard:
                    if card2level(card) in levels:
                        result.append(card)
                        levels.remove(card2level(card))
                yield result


def findStraight2(myCard):
    global counts
    for begin in range(0, 7):
        for end in range(begin + 5, 12):
            flag = True
            result = []
            for iter in range(begin, end + 1):
                if counts[iter] < 2:
                    flag = False
                    break
            if flag:
                levels = [iter for iter in range(begin, end + 1)] * 2
                for card in myCard:
                    if card2level(card) in levels:
                        result.append(card)
                        levels.remove(card2level(card))
                yield result


def findTriplet(myCard):
    global counts, Triplet, Bomb
    for triplet in Triplet:
        result = []
        for card in myCard:
            if card2level(card) == triplet:
                result.append(card)
        yield result
    for triplet in Bomb:
        result = []
        for card in myCard:
            if card2level(card) == triplet:
                result.append(card)
                if len(result) == 3:
                    yield result
                    continue


def findTriplet1(myCard):
    for each in findTriplet(myCard):
        tempCard = copy.copy(myCard)
        for card in each:
            tempCard.remove(card)
        for each2 in findSingle(tempCard):
            yield each + each2


def findTriplet2(myCard):
    for each in findTriplet(myCard):
        tempCard = copy.copy(myCard)
        for card in each:
            tempCard.remove(card)
        for each2 in findPair(tempCard):
            yield each + each2


def findBomb(myCard):
    global counts, Bomb
    for bomb in Bomb:
        result = []
        for card in myCard:
            if card2level(card) == bomb:
                result.append(card)
        yield result


def findQuadruple2(myCard):
    for each in findBomb(myCard):
        tempCard = copy.copy(myCard)
        for card in each:
            tempCard.remove(card)
        for each2 in findSingle(tempCard):
            tempCard2 = copy.copy(tempCard)
            for card in each2:
                tempCard2.remove(card)
            for each3 in findSingle(tempCard2):
                yield each + each2 + each3


def findQuadruple4(myCard):
    for each in findBomb(myCard):
        tempCard = copy.copy(myCard)
        for card in each:
            tempCard.remove(card)
        for each2 in findPair(tempCard):
            tempCard2 = copy.copy(tempCard)
            for card in each2:
                tempCard2.remove(card)
            for each3 in findPair(tempCard2):
                yield each + each2 + each3


def findPlane(myCard):
    global counts
    for begin in range(0, 12):
        for end in range(begin + 2, 14):
            flag = True
            result = []
            for iter in range(begin, end + 1):
                if counts[iter] < 3:
                    flag = False
                    break
            if flag:
                levels = [iter for iter in range(begin, end + 1)] * 3
                for card in myCard:
                    if card2level(card) in levels:
                        result.append(card)
                        levels.remove(card2level(card))
                yield result


def findPlane1(myCard):
    for each in findPlane(myCard):
        tempCard = copy.copy(myCard)
        for card in each:
            tempCard.remove(card)
        l = len(tempCard) // 3
        if l == 2:
            for each2 in findSingle(tempCard):
                tempCard2 = copy.copy(tempCard)
                for card in each2:
                    tempCard2.remove(card)
                for each3 in findSingle(tempCard2):
                    yield each + each2 + each3
        elif l == 3:
            for each2 in findSingle(tempCard):
                tempCard2 = copy.copy(tempCard)
                for card in each2:
                    tempCard2.remove(card)
                for each3 in findSingle(tempCard2):
                    tempCard3 = copy.copy(tempCard2)
                    for card in each3:
                        tempCard3.remove(card)
                    for each4 in findSingle(tempCard3):
                        yield each + each2 + each3 + each4


def findPlane2(myCard):
    for each in findPlane(myCard):
        tempCard = copy.copy(myCard)
        for card in each:
            tempCard.remove(card)
        l = len(tempCard) // 3
        if l == 2:
            for each2 in findPair(tempCard):
                tempCard2 = copy.copy(tempCard)
                for card in each2:
                    tempCard2.remove(card)
                for each3 in findPair(tempCard2):
                    yield each + each2 + each3
        elif l == 3:
            for each2 in findPair(tempCard):
                tempCard2 = copy.copy(tempCard)
                for card in each2:
                    tempCard2.remove(card)
                for each3 in findPair(tempCard2):
                    tempCard3 = copy.copy(tempCard2)
                    for card in each3:
                        tempCard3.remove(card)
                    for each4 in findPair(tempCard3):
                        yield each + each2 + each3 + each4


# def findSshuttle(myCard):
#     global counts
#     for begin in range(0, 12):
#         for end in range(begin + 2, 14):
#             flag = True
#             result = []
#             for iter in range(begin, end + 1):
#                 if counts[iter] < 4:
#                     flag = False
#                     break
#             if flag:
#                 levels = [iter for iter in range(begin, end + 1)] * 3
#                 for card in myCard:
#                     if card2level(card) in levels:
#                         result.append(card)
#                         levels.remove(card2level(card))
#                 yield result


# def findSshuttle2(myCard):
#     yield []


# def findSshuttle4(myCard):
#     yield []


def findRocket(myCard):
    if 52 in myCard and 53 in myCard:
        yield [52, 53]


# 分牌器
def findAllValid(myCard: list, lastValid: list):
    lastValid = cardCombo(lastValid)
    # 如果要管的牌是"过"
    if lastValid.combotype == "pass":
        # 尝试出带四条的
        for each in findBomb(myCard):
            yield each
        for each in findQuadruple2(myCard):
            yield each
        for each in findQuadruple4(myCard):
            yield each
        # for each in findSshuttle(myCard):
        #     yield each
        # for each in findSshuttle2(myCard):
        #     yield each
        # for each in findSshuttle4(myCard):
        #     yield each
        # 尝试出带三条的
        for each in findTriplet(myCard):
            yield each
        for each in findPlane(myCard):
            yield each
        for each in findTriplet1(myCard):
            yield each
        for each in findPlane1(myCard):
            yield each
        for each in findTriplet2(myCard):
            yield each
        for each in findPlane2(myCard):
            yield each
        # 尝试出对子
        for each in findPair(myCard):
            yield each
        for each in findStraight2(myCard):
            yield each
        # 尝试出单张
        for each in findSingle(myCard):
            yield each
        for each in findStraight(myCard):
            yield each
        for each in findRocket(myCard):
            yield each

    # 如果要管的牌不是"过"
    else:
        # 如果要管的是火箭直接过
        if lastValid.combotype == "rocket":
            yield []
            return
        # 否则按照牌型搜索能管得上的牌
        if lastValid.combotype == "single":
            for each in findSingle(myCard):
                if canBeat(each, lastValid):
                    yield each
        if lastValid.combotype == "pair":
            for each in findPair(myCard):
                if canBeat(each, lastValid):
                    yield each
        if lastValid.combotype == "straight":
            length = len(lastValid.packs)
            for i in range(0, 13 - length):
                flag = True
                for j in range(0, length):
                    if counts[i + j] == 0:
                        flag = False
                        break
                if flag:
                    result = []
                    levels = [k for k in range(i, i + length)]
                    for cards in myCard:
                        if card2level(cards) in levels:
                            result.append(cards)
                            levels.remove(card2level(cards))
                    yield result
        if lastValid.combotype == "straight2":
            length = len(lastValid.packs)
            for i in range(0, 13 - length):
                flag = True
                for j in range(0, length):
                    if counts[i + j] < 2:
                        flag = False
                        break
                if flag:
                    result = []
                    levels = [k for k in range(i, i + length)] * 2
                    for cards in myCard:
                        if card2level(cards) in levels:
                            result.append(cards)
                            levels.remove(card2level(cards))
                    yield result
        if lastValid.combotype == "triplet":
            for each in findTriplet(myCard):
                if canBeat(each, lastValid):
                    yield each
        if lastValid.combotype == "triplet1":
            for each in findTriplet1(myCard):
                if canBeat(each, lastValid):
                    yield each
        if lastValid.combotype == "triplet2":
            for each in findTriplet2(myCard):
                if canBeat(each, lastValid):
                    yield each
        if lastValid.combotype == "quadruple2":
            for each in findQuadruple2(myCard):
                if canBeat(each, lastValid):
                    yield each
        if lastValid.combotype == "quadruple4":
            for each in findQuadruple4(myCard):
                if canBeat(each, lastValid):
                    yield each
        if lastValid.combotype == "plane":
            for each in findPlane(myCard):
                if canBeat(each, lastValid):
                    yield each
        if lastValid.combotype == "plane1":
            for each in findPlane1(myCard):
                if canBeat(each, lastValid):
                    yield each
        if lastValid.combotype == "plane2":
            for each in findPlane2(myCard):
                if canBeat(each, lastValid):
                    yield each
        # if lastValid.combotype == "sshuttle":
        #     for each in findSshuttle(myCard):
        #         if canBeat(each, lastValid):
        #             yield each
        # if lastValid.combotype == "sshuttle2":
        #     for each in findSshuttle2(myCard):
        #         if canBeat(each, lastValid):
        #             yield each
        # if lastValid.combotype == "sshuttle4":
        #     for each in findSshuttle4(myCard):
        #         if canBeat(each, lastValid):
        #             yield each
        # 如果要管的不是炸弹，则搜索所有炸弹
        # 如果要管的是炸弹，则搜索能管得上的炸弹
        if lastValid.combotype != "bomb":
            findBomb(myCard)
        else:
            for each in findBomb(myCard):
                if canBeat(each, lastValid):
                    yield each
        # 火箭无论什么时候都可以出
        for each in findRocket(myCard):
            yield each
        # 最后则可以考虑过
        yield []


# 计算输入参数
def calculatePara(cards):
    global para1, para6, para7
    global myCard, cardRemaining
    tempCard = copy.copy(myCard)
    tempCardRemaining = copy.copy(cardRemaining)
    for card in cards:
        tempCard.remove(card)
    tempCardRemaining[myPosition] = len(tempCard)

    para3 = [0] * 18
    for j in range(0, 3):
        for k in range(0, 3):
            para3[j * 3 + k] = tempCardRemaining[j] * para1[k]
    for j in range(0, 9):
        para3[9 + j] = para3[j] ** 2

    para4 = cardAnalysis(tempCard)
    para5 = myAction(tempCard)

    return np.array(para1 + para3 + para4 + para5 + para6 + para7)


# 计算sigmoid函数值
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 计算概率
def calculateProb(para, theta1, theta2, theta3):
    h1 = sigmoid(np.hstack((np.array([1]), para)) @ theta1.transpose())
    h2 = sigmoid(np.hstack((np.array([1]), h1)) @ theta2.transpose())
    h3 = sigmoid(np.hstack((np.array([1]), h2)) @ theta3.transpose())
    return h3


# 复制自data-collect.py
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
        countOfCount = [0] * 5
        counts = [0] * 15
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
            self.combotype = "invalid"


def myAction(cards):
    para5 = [0] * 232
    myaction = cardCombo(cards)

    # acc = 0

    # 过
    if myaction.combotype == "pass":
        para5[0] = 1
        # acc += 1

    # 单张
    # 使用level的一次、两次和三次项
    if myaction.combotype == "single":
        para5[1 + myaction.combolevel] = 1
        # acc += 15

    # 对子
    # 使用level的一次、两次和三次项
    if myaction.combotype == "pair":
        para5[16 + myaction.combolevel] = 1
        # acc += 15

    # 顺子
    # 顺子起点的level有[0-7]共8种，终点有[4-11]共8种，共64种顺子
    if myaction.combotype == "straight":
        end = myaction.combolevel + findMaxSeq(myaction.packs) - 1
        para5[31 + myaction.combolevel * 8 + end - 4] = 1
        # acc += 64

    # 双顺
    # 双顺和单顺作同样处理
    if myaction.combotype == "straight2":
        end = myaction.combolevel + findMaxSeq(myaction.packs) - 1
        para5[95 + myaction.combolevel * 8 + end - 4] = 1
        # acc += 64

    # 三条
    # 因为纯三条比较少见，所以采用一次项和二次项
    if myaction.combotype == "triplet":
        para5[159] = myaction.combolevel
        para5[160] = myaction.combolevel
        # acc += 2

    # 三带一
    # 采用三条和单张的一次项
    if myaction.combotype == "triplet1":
        para5[161 + myaction.packs[0].level] = 1
        # acc += 15
        para5[176 + myaction.packs[1].level] = 1
        # acc += 15

    # 三带二
    # 与三带一类似
    if myaction.combotype == "triplet2":
        para5[191 + myaction.packs[0].level] = 1
        # acc += 15
        para5[206 + myaction.packs[1].level] = 1
        # acc += 15

    # 炸弹
    # 采用炸弹的一次项和二次项
    if myaction.combotype == "bomb":
        para5[221] = myaction.combolevel
        para5[222] = myaction.combolevel ** 2
        # acc += 2

    # 四带二（只）
    # 以下都比较稀少，只采用一个有或没有
    if myaction.combotype == "quadruple2":
        para5[223] = 1
        # acc += 1

    # 四带二（对）
    if myaction.combotype == "quadruple4":
        para5[224] = 1
        # acc += 1

    # 飞机
    if myaction.combotype == "plane":
        para5[225] = 1
        # acc += 1

    # 飞机带小翼
    if myaction.combotype == "plane1":
        para5[226] = 1
        # acc += 1

    # 飞机带大翼
    if myaction.combotype == "plane2":
        para5[227] = 1
        # acc += 1

    # 航天飞机
    if myaction.combotype == "sshuttle":
        para5[228] = 1
        # acc += 1

    # 航天飞机带小翼
    if myaction.combotype == "sshuttle2":
        para5[229] = 1
        # acc += 1

    # 航天飞机带大翼
    if myaction.combotype == "sshuttle4":
        para5[230] = 1
        # acc += 1

    # 火箭
    if myaction.combotype == "rocket":
        para5[231] = 1
        # acc += 1

    return para5


# 计算通用的参数
para1 = [0] * 3
para1[myPosition] = 1
para6 = myAction(lastValid)
para7 = [turn]


# 迭代计算最优解
pmax = 0
myOutput = []
# print(myCard)
# print(counts)
for each in findAllValid(myCard, lastValid):
    # print(each)
    para = calculatePara(each)
    p_curr = calculateProb(para, theta1, theta2, theta3)
    # print(p_curr)
    if p_curr > pmax:
        myOutput = copy.copy(each)
        pmax = p_curr

print(json.dumps({"response": myOutput}))

t1 = time.monotonic()
# print(t1 - t0)
