using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Antares
{
    // Deep Learning型ではない通常のMCTSで使用するノードクラス
    // 
    // n: このノードの試行回数
    // w: このノードの勝ち数
    // t: 親ノードの試行回数 = 子ノードの試行回数の総和
    // ※ 子ノード = このノードを含めた兄弟ノード
    //
    // 普通のUCB1 UCB1 = (w / n) + sqrt((2 * log(t) / n))

    // AlphaZero型のUCB1
    /*          float[] ucb1_array = new float[BTree.RootMoves.Length];
                t = 0;
                i = 1;
                while (i<BTree.RootMoves.Length)
                    t += NodeList[i++].PlayoutCount;
                i = 1;
                while (i<BTree.RootMoves.Length)
                {
                    float u = NodeList[i].PolicyResult * (float)Math.Sqrt(t) / (NodeList[i].PlayoutCount + 1);
                    float q = 0.0F;
                    if (NodeList[i].EvalCount > 0 && NodeList[i].PlayoutCount > 0)
                        q = (float) ((1 - value_lambda) * (NodeList[i].WinRateSum / NodeList[i].EvalCount) + value_lambda* (NodeList[i].WinCount / NodeList[i].PlayoutCount));
                    ucb1_array[i - 1] = u + q;
                    i++;
                }
                int max_index = Array.IndexOf(ucb1_array, ucb1_array.Max()) + 1;*/

    // 私が想定しているUCB1 Policy不使用版
    // UCB1 = q0 + q1
    // q0では特徴がKKP + KPPの評価関数を使用する。
    // q1では特徴がKKP + PP + 利きの評価関数を使用する。
    //
    // また、遷移確率の学習が成功したら、応用してみる。

    internal class Node
    {
        // 速度が出そうになかったのでアクセサやインデクサは意図的に使っていない
        public int color;
        public int ParentIndex;
        public int ThisIndex;
        public int TrialCount;
        public int PlayoutCount;
        //public int WinCount;
        //public int DrawCount;
       // public int LostCount;
        public int EvalCount;
        public float WinRateSum;
        public float LostRateSum;
        public float RewardSum;
        public bool IsLeaf;
        public List<int> ChildIndexes;
        public Move move;
        public float PolicyResult;
        public int Value;
        public Node()
        {
            color = 0;
            ParentIndex = int.MaxValue;
            ThisIndex = int.MaxValue;
            TrialCount = 0;
            PlayoutCount = 0;
            //WinCount = 0;
            //DrawCount = 0;
            //LostCount = 0;
            EvalCount = 0;
            WinRateSum = 0f;
            LostRateSum = 0f;
            RewardSum = 0f;
            IsLeaf = true;
            ChildIndexes = new List<int>();
            ChildIndexes.Clear();
            move = new Move();
            PolicyResult = 0f;
        }
    }

    internal class Node2
    {
        // 速度が出そうになかったのでアクセサやインデクサは意図的に使っていない
        public int color;
        public int ParentIndex;
        public int ThisIndex;
        public int TrialCount;
        public int PlayoutCount;
        public int WinCount;
        public int DrawCount;
        public int LostCount;
        public int EvalCount;
        public bool IsLeaf;
        public List<int> ChildIndexes;
        public Move move;
        //public float PolicyResult;
        //public int Value;
        public Node2()
        {
            color = 0;
            ParentIndex = int.MaxValue;
            ThisIndex = int.MaxValue;
            TrialCount = 0;
            PlayoutCount = 0;
            WinCount = 0;
            DrawCount = 0;
            LostCount = 0;
            EvalCount = 0;
            IsLeaf = true;
            ChildIndexes = new List<int>();
            ChildIndexes.Clear();
            move = new Move();
            //PolicyResult = 0f;
        }
    }

}
