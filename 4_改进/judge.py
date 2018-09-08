import decision
import sample
import sample2
import json
import random
import time


def main(iteration):
    t0 = time.monotonic()
    win = 0
    error = 0
    f = open("log.txt", "w")
    for j in range(0, iteration):
        t1 = time.monotonic()
        print("正在进行第", j, "局，用时", t1 - t0, "秒，", end="")
        # 首先分出身份
        # 0代表地主，1代表农民
        # myPosition = random.randint(0, 1)
        # otherPosition = 1 - myPosition

        # 然后分牌
        all_cards = [i for i in range(0, 54)]
        # 首先是地主的明牌
        publicCard = []
        for i in range(0, 3):
            tmprand = random.randint(0, len(all_cards) - 1)
            publicCard.append(all_cards[tmprand])
            all_cards.remove(all_cards[tmprand])
        # 然后是每个人的17张牌
        allocation = [[], [], []]
        for i in range(0, 3):
            while len(allocation[i]) < 17:
                tmprand = random.randint(0, len(all_cards) - 1)
                allocation[i].append(all_cards[tmprand])
                all_cards.remove(all_cards[tmprand])
        # 地主的牌加上三张明牌
        allocation[0] += publicCard

        # 初始化
        turns = 0
        history = [[[]], [[]], [[]]]
        others = [[1, 2], [2, 0], [0, 1]]
        response = [[], [], []]
        for i in range(0, 3):
            response[i] = [{"own": allocation[i],
                       "publiccard": publicCard}]
        cardRemaining = [20, 17, 17]
        log = []

        # 开始游戏
        run = True
        while run:
            try:
                # 循环进行游戏
                for i in range(0, 3):
                    if turns == 0:
                        response[i][0]["history"] = [history[others[i][0]][0], history[others[i][1]][0]]
                    else:
                        response[i].append({"history": [history[others[i][0]][-1], history[others[i][1]][-1]]})
                    line = {"requests": response[i],
                            "responses": history[i]}
                    if turns == 0:
                        line["responses"] = []
                    # print(json.dumps(line))

                    # if bool(myPosition) == bool(i):
                    #     try:
                    #         output = decision.main(json.dumps(line))
                    #     except:
                    #         raise ValueError("my bot")
                    # else:
                    #     try:
                    #         # output = sample.main(json.dumps(line))
                    #         output = sample2.main(json.dumps(line))
                    #     except:
                    #         raise ValueError("sample")
                    try:
                        output = decision.main(json.dumps(line))
                    except:
                        raise ValueError("my bot")
                    # line = {"response": output}
                    # print(json.dumps(line))
                    if turns == 0:
                        history[i][0] = output
                    else:
                        history[i].append(output)
                    cardRemaining[i] -= len(output)
                    log.append({"output": {"content": {str(i): {"history": response[i][-1]["history"]}}}})
                    log.append({str(i): {"response": output}})

                    # 判断游戏结束
                    if cardRemaining[0] == 0 or cardRemaining[1] == 0 or cardRemaining[2] == 0:
                        run = False
                        scores = [0, 0, 0]
                        # if myPosition == 0 and cardRemaining[0] == 0:
                        #     print("本局我胜利")
                        #     scores = [2.5, 0.1, 0.1]
                        #     win += 1
                        # elif myPosition == 1 and (cardRemaining[1] == 0 or cardRemaining[2] == 0):
                        #     print("本局我胜利")
                        #     scores = [0.1, 2.5, 2.5]
                        #     win += 1
                        # elif cardRemaining[0] == 0:
                        #     scores = [2.5, 0.1, 0.1]
                        #     print("本局sample2胜利")
                        # else:
                        #     scores = [0.1, 2.5, 2.5]
                        #     print("本局sample2胜利")
                        if cardRemaining[0] == 0:
                            print("本局地主胜利")
                            scores = [2.5, 0.1, 0.1]
                            win += 1
                        else:
                            print("本局农民胜利")
                            scores = [0.1, 2.5, 2.5]

                        # 写入log
                        initdata = {"publiccard": publicCard,
                                    "allocation": allocation}
                        line = {"initdata": initdata,
                                "scores": scores,
                                "log": log}
                        f.write(json.dumps(line))
                        f.write("\n")

                        break
                turns += 1
            except ValueError as ve:
                error += 1
                print("本局出现异常，异常来源于", ve.args[0])
                break
    print("地主的胜率是", win / (iteration - error))
    f.close()


if __name__ == '__main__':
    main(1000)
