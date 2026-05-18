using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Antares
{
    // 深層学習型のMCTS用ノードクラス
    internal class NodeDeep
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
        public float WinRateSum;
        public float LostRateSum;
        public bool IsLeaf;
        public List<int> ChildIndexes;
        public Move move;
        public float PolicyResult;

        public NodeDeep()
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
            WinRateSum = 0f;
            LostRateSum = 0f;
            IsLeaf = true;
            ChildIndexes = new List<int>();
            ChildIndexes.Clear();
            move = new Move();
            PolicyResult = 0f;
        }
    }
}
