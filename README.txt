檔案說明：

    RL_maze_pgm.py ：
        內容：
            使用強化學習、策略梯度法(Policy Gradient Method)找出迷宮最短路徑

        程式說明：
            1.使用策略迭代法中的策略梯度法(Policy Gradient Method)
            2.theta 轉換成 pi 使用 softmax 函數
                原因：
                    就算 theta 為負數，亦能算出機率。
                    (因為 exponential 輸出恆正)
            3.theta 更新公式：
                newTheta = oriTheta + eta * deltaTheta

                eta : 學習率
                deltaTheta :
                    deltaTheta = (Nij - Pij * Ni) / T
                        Nij : 在狀態為 si 的情況下，執行 action aj 的次數
                        Pij : 在目前策略下，狀態為 si 並執行 action aj 的機率
                        Ni : 狀態為 si 的情況下執行動作的次數
                        T : 總步數
            4.停止訓練的參數為 stop 變數，本程式將其設為 10^-4


    RL_maze_reward.py：
        內容：
            使用強化學習、價值迭代法、Sarsa 演算法找出迷宮最短路徑

        程式內容：
            1.使用價值迭代法
            2.Q 為行動價值函數
            3.使用 Sarsa 更新 Q
                Sarsa：
                    Q(st , at) = Q(st , at) + eta * (TD error)

                    TD error ( Temporal Deffierence error )：
                        TD error = Rt+1 + gamma * Q(st+1 , at+1) - Q(st , at)
            4.gamma 為時間折扣率 ( 概念類似銀行的複利率 )
            5.epsilon 為一機率，讓狀態根據此機率選擇隨機移動或是根據 Q函數移動。
            6.eta 為學習率
            7.count 為控制停止訓練模型的變數

    RL_maze_Q.py：
        內容：
            使用強化學習、價值迭代法、Q學習找出迷宮最短路徑

        程式內容：
            1.與 Sarsa 不同的只有 "更新 Q函數 的方式"
                Sarsa：
                    newQ[s , a] = Q[s , a] + eta * (r + gamma * Q[nextS , nextA] - Q[s , a])
                Q Learning：
                    newQ[s , a] = Q[s , a] + eta * (r + gamma * np.nanmax(Q[nextS,:]) - Q[s , a])

                Q Learning 採用下個狀態中具備「最大」行動價值的 Action

    RL_cartpole.py：
        內容：
            利用強化學習、價值迭代法、Q學習，解決「倒立單擺問題」

        程式內容：
            1. observation 包含四個數值：
                。車子位置 (-2.4 ~ 2.4 )
                。車子速度 (-Inf ~ Inf )
                。單擺角度 (-41.8 ~ 41.8 )
                。單擺角速度 (-Inf ~ Inf )
            2. 利用 cut 將數值離散化，CUT_SIZE 為切割的區塊數
            3. 每次執行 Action 後，進行 Q Table 的更新
            4. state ：
                (1) 將 observation 離散化 (切成6區塊，其數值疆界於 0 ~ 5)
                (2) 以 6 進為計算四個數值的和，結果極為 state
                    ex：
                        cartPos = 0, cartV = 4, poleAngle = 2, poleV = 5
                        state = 6**0 * cartPos +
                                6**1 * cartV +
                                6**2 * poleAngle +
                                6**3 * poleV
            5. action 只有兩種情況，即向左 (0) 以及向右 (1)
