using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Antares.AttacksOperation;
using static Antares.Board;
using static Antares.Common;
using static Antares.Evaluate;
using static Antares.GenMoves;
using static Antares.Mate1Ply;
using static Antares.Playout;
//using static TorchSharp.torch.distributions.constraints;
//using static TorchSharp.torch.nn;

namespace Antares
{
    // 深層学習型ではない通常のMCTSで使用するノードクラス
    // 学習した戦略と価値は使用しない。プレイアウトはランダム。

    internal class MCTS3
    {
        public BoardTree BTree;
        public List<Node2> NodeList = new List<Node2>();
        public int PlayOutCount;
        public float[] RootOutput;
        public int RootValue;
        public const float value_lambda = 0.5f;
        public const int nthr = 30;
        public int TaskNumber;
        public long SearchTimeLimit;// ミリ秒で指定する
        public const long TimeBuffer = 500;
        public Stopwatch sw = new Stopwatch();
        public Queue<string> queue_from_main_thread = new Queue<string>();
        public Queue<string> queue_to_main_thread = new Queue<string>();
        public TT tt = new TT();
        public bool is_abort;
        public bool is_finished;

        public void GenRootMoves()
        {
            List<Move> move_list = new List<Move>();
            if (IsAttacked(BTree, BTree.SQ_King[(int)BTree.RootColor], (int)BTree.RootColor) == 0)
            {
                GenCap(BTree, (int)BTree.RootColor, ref move_list);
                GenNoCap(BTree, (int)BTree.RootColor, ref move_list);
                GenDrop(BTree, (int)BTree.RootColor, ref move_list);
            }
            else
            {
                GenEvasion(BTree, (int)BTree.RootColor, ref move_list);
            }

            BTree.RootMoves = move_list.ToArray();// 玉を取る非合法手とDiscoverd Checkの手を除去する処理は入っていない。
        }

        public void Root()
        {
            //float[] result_array = new float[BTree.RootMoves.Length];

            NodeList.Clear();

            Node2 root_node = new Node2();
            NodeList.Add(root_node);

            RootValue = (BTree.RootColor == Antares.Common.Color.Black) ? Eval(BTree) : -Eval(BTree);
            BTree.EvalArray[BTree.ply] = RootValue;

            for (int i = 0; i < BTree.RootMoves.Length; i++)// 要対応：RootMovesは全合法手を展開するか否か？
            {
                Node2 n = new Node2();
                n.color = (int)BTree.RootColor;
                Move m = BTree.RootMoves[i];
                n.ParentIndex = 0;
                n.ThisIndex = i + 1;
                n.move = m;
                NodeList[0].ChildIndexes.Add(i + 1);
                //Do(ref BTree, m, (int)BTree.RootColor);
                //n.Value = -EvalWrapper(BTree, (int)BTree.RootColor, 1, m, false);
                //var aaa = (BTree.RootColor == Antares.Common.Color.Black) ? -Eval(BTree) : Eval(BTree);
                //UnDo(ref BTree, m, (int)BTree.RootColor);
                //float f = (float)n.Value / 600;
                //n.PolicyResult = Sigmoid((float)n.Value / 600);// 後手番の場合の反転漏れを確認する。
                //n.PolicyResult = RootOutput[i];
                //result_array[i] = RootOutput[i];
                NodeList.Add(n);
            }

            while (true)
            {
                long elapsed = sw.ElapsedMilliseconds;
                if (SearchTimeLimit < elapsed)
                    break;

                if (is_abort)
                    break;

                int t, i;
                float[] ucb1_array = new float[BTree.RootMoves.Length];
                t = 0;
                i = 1;
                while (i < BTree.RootMoves.Length)
                    t += NodeList[i++].PlayoutCount;
                i = 1;
                while (i < BTree.RootMoves.Length + 1)
                {
                    //float u = NodeList[i].PolicyResult * (float)Math.Sqrt(t) / (NodeList[i].PlayoutCount + 1);
                    //float u = NodeList[i].PolicyResult * (float)Math.Sqrt(t) / (NodeList[i].PlayoutCount + 1);
                    //float q = 0.0F;
                    //if (NodeList[i].EvalCount > 0 && NodeList[i].PlayoutCount > 0)
                    //q = (float)(1 - value_lambda) * (NodeList[i].WinRateSum / NodeList[i].EvalCount) + value_lambda * (NodeList[i].RewardSum / NodeList[i].PlayoutCount);
                    //q = (float)((1 - value_lambda) * (NodeList[i].WinRateSum / NodeList[i].EvalCount) + value_lambda * (NodeList[i].WinCount / NodeList[i].PlayoutCount));
                    //ucb1_array[i - 1] = u + q;
                    if (t == 0)
                        t = 1;
                    ucb1_array[i - 1] = NodeList[i].WinCount / (NodeList[i].PlayoutCount + 1) + (float)Math.Sqrt((2 * (float)Math.Log(t) / (NodeList[i].PlayoutCount + 1)));
                    i++;
                }

                int max_index = Array.IndexOf(ucb1_array, ucb1_array.Max()) + 1;
                Do(ref BTree, BTree.RootMoves[max_index - 1], (int)BTree.RootColor);

                // 指した手によって自玉がDiscovered Checkになってしまった場合
                /*if (IsAttacked(BTree, BTree.SQ_King[(int)BTree.RootColor], (int)BTree.RootColor) != 0)
                {
                    UnDo(ref BTree, (int)BTree.RootColor, BTree.RootMoves[max_index - 1]);
                    continue;
                }*/

                // 玉を取ってしまう非合法手の場合
                /*if (BTree.RootMoves[max_index - 1].CapPiece == Piece.Type.King)
                {
                    UnDo(ref BTree, (int)BTree.RootColor, BTree.RootMoves[max_index - 1]);
                    continue;
                }*/

                if (NodeList[max_index].IsLeaf)
                {
                    elapsed = sw.ElapsedMilliseconds;
                    if (SearchTimeLimit < (elapsed + TimeBuffer))
                        break;
                    if (is_abort)
                        break;

                    if (NodeList[max_index].TrialCount >= nthr)
                    {
                        ExpandNode(((int)BTree.RootColor ^ 1), max_index, BTree.ply);
                        if (is_abort)
                            break;
                    }
                    else
                    {
                        BoardTree bt = new BoardTree();
                        BoardTreeAlloc(ref bt);
                        bt = DeepCopy(BTree, false);
                        List<Move> moves = new List<Move>();
                        //int v = EvalWrapper(bt, (int)bt.RootColor, bt.ply, BTree.RootMoves[max_index - 1], false);
                        // ToDo:  vの反転漏れをチェックする。
                        //int v = NodeList[max_index].Value;
                        //float result = PlayOutShallow(ref bt, ((int)BTree.RootColor ^ 1), BTree.ply, ref moves, NodeList[max_index].Value);
                        int result = PlayOutNoSearch(ref bt, ((int)BTree.RootColor ^ 1), BTree.ply, ref moves);
                        NodeList[max_index].PlayoutCount++;
                        NodeList[max_index].EvalCount++;
                        //EvalNode(max_index, ((int)BTree.RootColor ^ 1), v);
                        if (is_abort)
                            break;
                        UpdateParam(max_index, result);
                    }
                }
                else
                {
                    elapsed = sw.ElapsedMilliseconds;
                    if (SearchTimeLimit < elapsed + TimeBuffer)
                        break;
                    if (is_abort)
                        break;
                    DescendNode((int)BTree.RootColor ^ 1, max_index, nthr, BTree.ply + 1);
                }

                UnDo(ref BTree, BTree.RootMoves[max_index - 1], (int)BTree.RootColor);
            }

        }

        private static float Sigmoid(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }

        private void ExpandNode(int color, int parent_index, int ply)
        {
            // 1手詰めと宣言勝ちをチェックし、勝ちがあった場合は手番側の勝ちをNodeに設定してreturnする color = 1
            //Move[] mate_move = new Move[3];
            Move mate_move = new Move();
            UInt128 ulret = IsAttacked(BTree, BTree.SQ_King[color], color);
            if (ulret == 0 && MateIn1Ply(BTree, color).Value != 0)
            {
                //NodeList[parent_index].WinRateSum = 0.0F;
                //NodeList[parent_index].LostRateSum = 1.0F;
                NodeList[parent_index].WinCount += 1;
                NodeList[parent_index].EvalCount += 1;
                return;
            }
            else
            {
                if (IsAttacked(BTree, BTree.SQ_King[color ^ 1], color ^ 1) != 0)
                {
                    int iret = IsDeclarationWin(BTree);
                    if (iret == color + 1)
                    {
                        //NodeList[parent_index].WinRateSum = 0.0F;
                        //NodeList[parent_index].LostRateSum = 1.0F;
                        NodeList[parent_index].LostCount += 1;
                        NodeList[parent_index].EvalCount += 1;
                        return;
                    }
                    else if (color == 0 && iret == 2 || color == 1 && iret == 1)
                    {
                        //NodeList[parent_index].WinRateSum = 1.0F;
                        //NodeList[parent_index].LostRateSum = 0.0F;
                        NodeList[parent_index].WinCount += 1;
                        NodeList[parent_index].EvalCount += 1;
                        return;
                    }
                }
            }

            // 自玉に王手がかかっている場合
            if (ulret != 0)
            {
                List<Move> evasion_moves = new List<Move>();
                bool flag2 = false;
                try
                {
                    GenEvasion(BTree, color, ref evasion_moves);
                }
                catch { flag2 = true; }
                // 王手を逃れる手が生成できなかったら詰み
                if (evasion_moves.Count == 0 || flag2 == true)
                {
                    //NodeList[parent_index].WinRateSum = 1.0F;
                    //NodeList[parent_index].LostRateSum = 0.0F;
                    NodeList[parent_index].WinCount += 1;
                    NodeList[parent_index].EvalCount += 1;
                    return;
                }
            }

            int current_index;
            bool flag = false;

            //if (!tt.PolicyMoves.ContainsKey(BTree.CurrentHash))
            {
                List<Move> moves = new List<Move>();

                if (IsAttacked(BTree, BTree.SQ_King[color], color) == 0)
                {
                    GenCap(BTree, color, ref moves);
                    GenNoCap(BTree, color, ref moves);
                    GenDrop(BTree, color, ref moves);
                }
                else
                {
                    GenEvasion(BTree, color, ref moves);
                }

                List<Move> legal_moves = new List<Move>();

                for (int i = 0; i < moves.Count; i++)
                {
                    if (IsMoveValid(ref BTree, moves[i], color))
                        legal_moves.Add(moves[i]);
                }

                for (int i = 0; i < legal_moves.Count; i++)
                {
                    Node2 n = new Node2();
                    n.color = color;
                    //string[] s2 = s[i].Split(" ");
                    Move m = legal_moves[i];

                    // 自玉がDiscovered Checkになってしまった場合
                    if (IsAttacked(BTree, BTree.SQ_King[color], color) > 0)
                    {
                        //UnDo(ref BTree, m, color);
                        continue;
                    }

                    int ifrom = m.From;
                    int ito = m.To;
                    int icap_pc = (int)m.CapPiece;

                    if (ifrom < Square_NB)
                    {
                        // PIN駒を動かしてしまった場合
                        Direction idirec = Adirec[ifrom, ito];
                        if (IsPinnedOnKing(BTree, ifrom, idirec, color) > 0)
                        {
                            continue;
                        }
                    }
                    else
                    {
                        // 駒打ちなのに駒取りになってしまった場合 = > 多分発生しないが一応入れておく
                        if (icap_pc != (int)Piece.Empty)
                        {
                            continue;
                        }
                    }

                    // 玉を取ってしまった場合
                    if (icap_pc == (int)Piece.King)
                    {
                        continue;
                    }


                    try
                    {
                        Do(ref BTree, m, color);
                    }
                    catch { continue; }

                    // 指した手によって自玉がDiscovered Checkになってしまった場合
                    if (IsAttacked(BTree, BTree.SQ_King[color], color) != 0)
                    {
                        UnDo(ref BTree, m, color);
                        continue;
                    }

                    n.ThisIndex = NodeList.Count;
                    n.ParentIndex = NodeList[parent_index].ThisIndex;
                    n.move = m;
                    //moves.Add(m);

                    //Do(ref BTree, color, m);

                    NodeList[parent_index].ChildIndexes.Add(n.ThisIndex);
                    //n.PolicyResult = float.Parse(s2[1]);// SoftMax関数はPython側でかける
                    //tt.PolicyDict.TryAdd(BTree.CurrentHash, n.PolicyResult);
                    current_index = n.ThisIndex;
                    BoardTree bt = new BoardTree();
                    BoardTreeAlloc(ref bt);
                    bt = DeepCopy(BTree, false);

                    List<Move> ms = new List<Move>();
                    int result = PlayOutNoSearch(ref bt, color ^ 1, ply + 1, ref ms);
                    //n.PolicyResult = Sigmoid((float)n.Value / 600);// 後手番の場合の反転漏れを確認する。
                    NodeList.Add(n);
                    //EvalNode(current_index, color ^ 1, v);
                    NodeList[current_index].PlayoutCount++;
                    NodeList[current_index].EvalCount++;
                    UpdateParam(current_index, result);

                    UnDo(ref BTree, m, color);

                    flag = true;
                }
                //tt.PolicyMoves.TryAdd(BTree.CurrentHash, moves);
            }

            if (flag == true)
                NodeList[parent_index].IsLeaf = false;
        }

        private void UpdateParam(int node_index, int result)
        {
            NodeList[node_index].TrialCount += 1;
            //NodeList[node_index].RewardSum += (float)result;

            if (result == 0)
            {
                if (NodeList[node_index].color == (int)BTree.RootColor)
                {
                    NodeList[node_index].WinCount += 1;
                }
                else
                {
                    NodeList[node_index].LostCount += 1;
                }
            }
            else if (result == 1)
            {
                if (NodeList[node_index].color == (int)BTree.RootColor)
                {
                    NodeList[node_index].LostCount += 1;
                }
                else
                {
                    NodeList[node_index].WinCount += 1;
                }
            }
            else
            {
                NodeList[node_index].DrawCount += 1;
            }
            Node2 current_node = NodeList[node_index];
            if (current_node.ParentIndex == 0)
                return;
            while (true)
            {
                int index = current_node.ParentIndex;
                current_node = NodeList[index];
                NodeList[index].TrialCount += 1;
                NodeList[index].PlayoutCount += 1;
                if (result == 0)
                {
                    if (NodeList[index].color == (int)BTree.RootColor)
                    {
                        NodeList[index].WinCount += 1;
                    }
                    else
                    {
                        NodeList[index].LostCount += 1;
                    }
                }
                else if (result == 1)
                {
                    if (NodeList[index].color == (int)BTree.RootColor)
                    {
                        NodeList[index].LostCount += 1;
                    }
                    else
                    {
                        NodeList[index].WinCount += 1;
                    }
                }
                else
                {
                    NodeList[index].DrawCount += 1;
                }

                NodeList[index].EvalCount += 1;
                if (current_node.ParentIndex == 0)
                    break;
            }
        }

        private void DescendNode(int color, int node_index, int nthr, int ply)
        {
            if (NodeList[node_index].ChildIndexes.Count == 0)
                return;

            int idx = new int();
            bool flag = false;
            while (true)
            {
                long elapsed = sw.ElapsedMilliseconds;
                if (SearchTimeLimit < elapsed + TimeBuffer)
                    break;
                if (is_abort)
                    break;
                int t, i;
                float[] ucb1_array = new float[NodeList[node_index].ChildIndexes.Count];
                t = 0;
                i = 0;
                while (i < NodeList[node_index].ChildIndexes.Count)
                {
                    idx = NodeList[node_index].ChildIndexes[i];
                    t += NodeList[idx].PlayoutCount;
                    i++;
                }

                i = 0;
                while (i < NodeList[node_index].ChildIndexes.Count)
                {
                    idx = NodeList[node_index].ChildIndexes[i];
                    //float u = NodeList[idx].PolicyResult * (float)Math.Sqrt(t) / (NodeList[idx].PlayoutCount + 1);
                    //float u = NodeList[idx].WinCount / (NodeList[idx].PlayoutCount + 1) 
                    //float q = 0.0F;
                    //if (NodeList[idx].EvalCount > 0 && NodeList[idx].PlayoutCount > 0)
                    //q = (float)(NodeList[i].WinRateSum / NodeList[i].EvalCount);
                    //q = (float)((1 - value_lambda) * (NodeList[idx].WinRateSum / NodeList[idx].EvalCount) + value_lambda * (NodeList[idx].WinCount / NodeList[idx].PlayoutCount));
                    ucb1_array[i++] = NodeList[idx].WinCount / (NodeList[idx].PlayoutCount + 1) + (float)Math.Sqrt((2 * (float)Math.Log(t) / (NodeList[idx].PlayoutCount + 1)));
                }

                int max_index = Array.IndexOf(ucb1_array, ucb1_array.Max());
                idx = NodeList[node_index].ChildIndexes[max_index];
                Do(ref BTree, NodeList[idx].move, color);

                if (NodeList[idx].IsLeaf)
                {
                    elapsed = sw.ElapsedMilliseconds;
                    if (SearchTimeLimit < elapsed + TimeBuffer)
                    {
                        flag = true;
                        goto end;
                    }

                    if (is_abort)
                        break;

                    if (NodeList[idx].TrialCount >= nthr)
                    {
                        ExpandNode(color ^ 1, idx, ply + 1);
                        if (is_abort)
                            break;
                    }
                    else
                    {
                        BoardTree bt = new BoardTree();
                        BoardTreeAlloc(ref bt);
                        bt = DeepCopy(BTree, false);
                        List<Move> moves = new List<Move>();
                        //int v = NodeList[max_index].Value;
                        //float result = PlayOutShallow(ref bt, color ^ 1, ply + 1, ref moves, v);
                        int result = PlayOutNoSearch(ref bt, color ^ 1, ply + 1, ref moves);
                        NodeList[idx].PlayoutCount++;
                        PlayOutCount++;
                        NodeList[idx].EvalCount++;
                        //EvalNode(idx, color ^ 1, v);
                        if (is_abort)
                            break;
                        UnDo(ref BTree, NodeList[idx].move, color);
                        AscendNode(color ^ 1, idx, result);
                        return;
                    }
                }
                else
                {
                    elapsed = sw.ElapsedMilliseconds;
                    if (SearchTimeLimit < elapsed + TimeBuffer)
                    {
                        flag = true;
                        goto end;
                    }

                    if (is_abort)
                        break;

                    DescendNode(color ^ 1, idx, nthr, ply + 1);
                    return;
                }

                UnDo(ref BTree, NodeList[idx].move, color);

            end:

                if (flag)
                {
                    Node2 current_node = NodeList[idx];
                    int temp_color = color ^ 1;
                    while (true)
                    {
                        int index = current_node.ParentIndex;
                        current_node = NodeList[index];
                        if (current_node.ParentIndex == 0)
                            break;
                        UnDo(ref BTree, current_node.move, temp_color);
                        temp_color ^= 1;
                    }
                    break;
                }
            }
        }

        // ※ まだ修正していない。
        void AscendNode(int color, int node_index, float result)// パラメータ plyは要らないか？
        {
            NodeList[node_index].TrialCount += 1;
            if (result == 0)
            {
                if (NodeList[node_index].color == (int)BTree.RootColor)
                {
                    NodeList[node_index].WinCount += 1;
                }
                else
                {
                    NodeList[node_index].LostCount += 1;
                }
            }
            else if (result == 1)
            {
                if (NodeList[node_index].color == (int)BTree.RootColor)
                {
                    NodeList[node_index].LostCount += 1;
                }
                else
                {
                    NodeList[node_index].WinCount += 1;
                }
            }
            else
            {
                NodeList[node_index].DrawCount += 1;
            }
            Node2 current_node = NodeList[node_index];
            if (current_node.ParentIndex == 0)
                return;
            int temp_color = color;
            //int temp_ply = ply - 1;
            while (true)
            {
                int index = current_node.ParentIndex;
                current_node = NodeList[index];
                NodeList[index].TrialCount += 1;
                NodeList[index].PlayoutCount += 1;
                if (result == 0)
                {
                    if (NodeList[index].color == (int)BTree.RootColor)
                    {
                        NodeList[index].WinCount += 1;
                    }
                    else
                    {
                        NodeList[index].LostCount += 1;
                    }
                }
                else if (result == 1)
                {
                    if (NodeList[index].color == (int)BTree.RootColor)
                    {
                        NodeList[index].LostCount += 1;
                    }
                    else
                    {
                        NodeList[index].WinCount += 1;
                    }
                }
                else
                {
                    NodeList[index].DrawCount += 1;
                }

                NodeList[index].EvalCount += 1;
                if (current_node.ParentIndex == 0)
                    break;
                UnDo(ref BTree, current_node.move, temp_color);
                temp_color ^= 1;
            }
        }
    }
}
