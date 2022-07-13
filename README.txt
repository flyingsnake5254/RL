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
            4.gamma 為時間折扣率 ( 概念類似銀行的複利率 )
            5.epsilon 為一機率，讓狀態根據此機率選擇隨機移動或是根據 Q函數移動。
            6.eta 為學習率
            7.count 為控制停止訓練模型的變數