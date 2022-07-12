檔案說明：

RL_maze_pgm.py：
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