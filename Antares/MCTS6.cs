using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static Antares.AttacksOperation;
using static Antares.Board;
using static Antares.Common;
using static Antares.Evaluate3;
using static Antares.GenMoves;
using static Antares.Mate1Ply;
using static Antares.Playout;
using static Antares.TransProbs6;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions.constraints;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim.lr_scheduler.impl.CyclicLR;

namespace Antares
{
    // 深層学習型ではない通常のMCTSで使用するノードクラス
    // ただし学習した評価関数を使う。

    // ※ ply >= 90の場合はプレイアウトを実施する方が良いかもしれない。
    // Policyには遷移確率を使用する。

    // MCTS2と同等だが、評価関数がHalfなのに注意。

    // これも現状性能が低調なので不使用です…。

    internal class MCTS6
    {
        public BoardTree BTree;
        public List<Node> NodeList = new List<Node>();
        public Move[] current_move = new Move[Ply_Max];
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
        const int input_num = 1170;
        const int middle_num = 32;
        const int output_num = 1;
        //public torch.Device d = torch.device(DeviceType.CPU);
        //public Sequential seq = Sequential(("input_layer", Linear(input_num, middle_num)), ("input_norm", BatchNorm1d(middle_num)), ("input_relu", ReLU()), ("output_layer", Linear(middle_num, output_num)), ("output_sigmoid", torch.nn.Sigmoid()));
        public static Device d = torch.device(DeviceType.CPU);
        public static TorchSharp.Modules.Linear input_layer = Linear(input_num, middle_num, device: d);
        public static TorchSharp.Modules.BatchNorm1d input_norm = BatchNorm1d(middle_num, 2e-05, device: d);
        public static TorchSharp.Modules.Linear output_layer = Linear(middle_num, output_num, device: d);
        public static TorchSharp.Modules.Sequential seq = Sequential(("input_layer", input_layer), ("input_norm", input_norm), ("input_relu", ReLU()), ("output_layer", output_layer), ("output_sigmoid", nn.Sigmoid()));


        public void LoadModel()
        {
            seq.load("model_tp.pth");
            seq.eval();
        }

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

        public List<Move> TotalParam(int num_tasks, ref List<float> f, ref List<int> t, ref List<MCTS6> mcts_tree)
        {
            int value_max, max_index, limit;
            List<int> trial_count_array = new List<int>();
            List<float> win_rate_array = new List<float>();
            List<Move> return_moves = new List<Move>();
            List<float> return_win_rate_array = new List<float>();
            List<int> return_trial_count_array = new List<int>();
            MCTS6 temp_tree = new MCTS6();
            BoardTree BTree = mcts_tree[0].BTree;

            switch (BTree.RootMoves.Length)
            {
                case 1:
                    limit = 1;
                    break;
                case 2:
                    limit = 2;
                    break;
                default:
                    limit = 3;
                    break;
            }

            if (num_tasks > 1)
            {
                for (int i = 1; i < num_tasks; i++)
                {
                    temp_tree = mcts_tree[i];
                    if (temp_tree.NodeList.Count < 2)
                        continue;
                    for (int j = 0; j < BTree.RootMoves.Length; j++)
                    {
                        mcts_tree[0].NodeList[j + 1].TrialCount += temp_tree.NodeList[j + 1].TrialCount;
                        //mcts_tree[0].NodeList[j + 1].WinCount += temp_tree.NodeList[j + 1].WinCount;
                        //mcts_tree[0].NodeList[j + 1].DrawCount += temp_tree.NodeList[j + 1].DrawCount;
                        //mcts_tree[0].NodeList[j + 1].LostCount += temp_tree.NodeList[j + 1].LostCount;
                        mcts_tree[0].NodeList[j + 1].EvalCount += temp_tree.NodeList[j + 1].EvalCount;
                        mcts_tree[0].NodeList[j + 1].RewardSum += temp_tree.NodeList[j + 1].RewardSum;
                        mcts_tree[0].NodeList[j + 1].WinRateSum += temp_tree.NodeList[j + 1].WinRateSum;
                        mcts_tree[0].NodeList[j + 1].LostRateSum += temp_tree.NodeList[j + 1].LostRateSum;
                    }
                }
            }

            for (int i = 0; i < BTree.RootMoves.Length; i++)
            {
                trial_count_array.Add(mcts_tree[0].NodeList[i + 1].TrialCount);
            }

            for (int i = 0; i < BTree.RootMoves.Length; i++)
            {
                if (trial_count_array[i] == 0)
                {
                    win_rate_array.Add(0);
                }
                else
                {
                    //win_rate_array.Add((float)(mcts_tree[0].NodeList[i + 1].RewardSum / (float)(trial_count_array[i])));
                    win_rate_array.Add((float)(mcts_tree[0].NodeList[i + 1].RewardSum));
                }
            }

            int max_node_index = 0;
            for (int i = 0; i < limit; i++)
            {
                value_max = trial_count_array.Max();
                max_index = Array.IndexOf(trial_count_array.ToArray(), value_max);
                if (i == 0)
                    max_node_index = max_index + 1;
                return_trial_count_array.Add(trial_count_array[max_index]);
                trial_count_array[max_index] = int.MinValue;
                return_moves.Add(BTree.RootMoves[max_index]);
                return_win_rate_array.Add(win_rate_array[max_index]);
            }

            Move best_move = return_moves[0];
            Node best_move_node = mcts_tree[0].NodeList[max_node_index];
            Node current_node = new Node();
            Node node = new Node();
            int depth = 1;
            int index, mi, mv;
            if (best_move_node.ChildIndexes.Count > 0)
            {
                int[] temp_trial_count_array = new int[best_move_node.ChildIndexes.Count];
                for (int i = 0; i < best_move_node.ChildIndexes.Count; i++)
                {
                    index = best_move_node.ChildIndexes[i];
                    current_node = mcts_tree[0].NodeList[index];
                    temp_trial_count_array[i] = current_node.TrialCount;
                }
                mv = temp_trial_count_array.Max();
                mi = Array.IndexOf(temp_trial_count_array, mv) + 1;
                //pv.Add(mcts_tasks[0].NodeList[mi].move);
                node = mcts_tree[0].NodeList[mi];
                while (true)
                {
                    if (node.ChildIndexes.Count == 0)
                        break;
                    temp_trial_count_array = new int[node.ChildIndexes.Count];
                    for (int i = 0; i < node.ChildIndexes.Count; i++)
                    {
                        index = node.ChildIndexes[i];
                        current_node = mcts_tree[0].NodeList[index];
                        temp_trial_count_array[i] = current_node.TrialCount;
                    }
                    mv = temp_trial_count_array.Max();
                    mi = Array.IndexOf(temp_trial_count_array, mv);
                    node = mcts_tree[0].NodeList[node.ChildIndexes[mi]];
                    depth++;
                }
            }

            f = return_win_rate_array;
            t = return_trial_count_array;

            return return_moves;
        }

        public void Root()
        {
            //float[] result_array = new float[BTree.RootMoves.Length];

            //Console.WriteLine("root_0");

            NodeList.Clear();

            Node root_node = new Node();
            NodeList.Add(root_node);

            RootValue = (BTree.RootColor == Antares.Common.Color.Black) ? (int)Eval(BTree) : -(int)Eval(BTree);
            BTree.EvalArray[BTree.ply] = RootValue;
            int[] colors = new int[BTree.RootMoves.Length];
            Move[] prev_moves = new Move[BTree.RootMoves.Length];

            if (BTree.ply == 1)
            {
                for (int i = 0; i < BTree.RootMoves.Length; i++)
                    prev_moves[i] = new Move();
            }
            else
            {
                for (int i = 0; i < BTree.RootMoves.Length; i++)
                    prev_moves[i] = current_move[BTree.ply - 1];
            }

            for (int i = 0; i < BTree.RootMoves.Length; i++)// 要対応：RootMovesは全合法手を展開するか否か？
            {
                Node n = new Node();
                n.color = (int)BTree.RootColor;
                Move m = BTree.RootMoves[i];
                n.ParentIndex = 0;
                n.ThisIndex = i + 1;
                n.move = m;
                NodeList[0].ChildIndexes.Add(i + 1);
                Do(ref BTree, m, (int)BTree.RootColor);
                n.Value = -(int)EvalWrapper(BTree, (int)BTree.RootColor, 1, m, false);
                //var aaa = (BTree.RootColor == Antares.Common.Color.Black) ? -Eval(BTree) : Eval(BTree);
                UnDo(ref BTree, m, (int)BTree.RootColor);
                //float f = (float)n.Value / 600;
                //n.PolicyResult = Sigmoid((float)n.Value / 600);
                //n.PolicyResult = RootOutput[i];
                //result_array[i] = RootOutput[i];
                NodeList.Add(n);
                colors[i] = (int)BTree.RootColor;
            }

            List<double[]> li_ft = MakeInputFeatures(BTree, colors, BTree.RootMoves.ToList(), prev_moves);
            torch.Tensor x = torch.zeros(BTree.RootMoves.Length, input_num, dtype: torch.float32, device: d);
            var ar_ft = li_ft.ToArray();
            for (int i = 0; i < ar_ft.Length; i++)
                x[i] = torch.tensor(ar_ft[i]);
            var y = seq.forward(x);
            List<double> li_p = new List<double>();
            //double sum_y = y.sum().ToDouble();
            for (int i = 0; i < ar_ft.Length; i++)
            {
                double dv = y[i].ToDouble();
                li_p.Add(dv);
                NodeList[i + 1].PolicyResult = (float)dv;
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
                while (i < BTree.RootMoves.Length)
                {
                    //float u = NodeList[i].PolicyResult * (float)Math.Sqrt(t) / (NodeList[i].PlayoutCount + 1);
                    float u = NodeList[i].PolicyResult * (float)Math.Sqrt(t) / (NodeList[i].PlayoutCount + 1);
                    float q = 0.0F;
                    if (NodeList[i].EvalCount > 0 && NodeList[i].PlayoutCount > 0)
                        q = (float)(1 - value_lambda) * (NodeList[i].WinRateSum / NodeList[i].EvalCount) + value_lambda * (NodeList[i].RewardSum / NodeList[i].PlayoutCount);
                    //q = (float)((1 - value_lambda) * (NodeList[i].WinRateSum / NodeList[i].EvalCount) + value_lambda * (NodeList[i].WinCount / NodeList[i].PlayoutCount));
                    //ucb1_array[i - 1] = u + q;
                    ucb1_array[i - 1] = u + q;
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
                        //Console.WriteLine("root_loop0");
                        ExpandNode(((int)BTree.RootColor ^ 1), max_index, BTree.ply);
                        if (is_abort)
                            break;
                        //Console.WriteLine("root_loop1");
                    }
                    else
                    {
                        //Console.WriteLine("root_loop2");
                        BoardTree bt = new BoardTree();
                        BoardTreeAlloc(ref bt);
                        bt = DeepCopy(BTree, false);
                        //bool b = Test.CompareBoard(BTree, bt);
                        List<Move> moves = new List<Move>();
                        //int v = EvalWrapper(bt, (int)bt.RootColor, bt.ply, BTree.RootMoves[max_index - 1], false);
                        // ToDo:  vの反転漏れをチェックする。
                        int v = NodeList[max_index].Value;
                        float result = PlayOutShallow(ref bt, ((int)BTree.RootColor ^ 1), BTree.ply, ref moves, NodeList[max_index].Value);
                        NodeList[max_index].PlayoutCount++;
                        //Console.WriteLine("root_loop3");
                        EvalNode(max_index, ((int)BTree.RootColor ^ 1), v);
                        if (is_abort)
                            break;
                        UpdateParam(max_index, result);
                        //Console.WriteLine("root_loop4");
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
            //if (m3p.IsMateIn3Ply(ref BTree, color, true, ref mate_move))
            UInt128 ulret = IsAttacked(BTree, BTree.SQ_King[color], color);
            if (ulret == 0 && MateIn1Ply(BTree, color).Value != 0)
            {
                NodeList[parent_index].WinRateSum = 0.0F;
                NodeList[parent_index].LostRateSum = 1.0F;
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
                        NodeList[parent_index].WinRateSum = 0.0F;
                        NodeList[parent_index].LostRateSum = 1.0F;
                        NodeList[parent_index].EvalCount += 1;
                        return;
                    }
                    else if (color == 0 && iret == 2 || color == 1 && iret == 1)
                    {
                        NodeList[parent_index].WinRateSum = 1.0F;
                        NodeList[parent_index].LostRateSum = 0.0F;
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
                    NodeList[parent_index].WinRateSum = 1.0F;
                    NodeList[parent_index].LostRateSum = 0.0F;
                    NodeList[parent_index].EvalCount += 1;
                    return;
                }
            }

            int current_index;
            bool flag = false;
            int start_index = 0;
            List<Move> legal_moves = new List<Move>();


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

                int nodes_added = 0;

                for (int i = 0; i < moves.Count; i++)
                {
                    Node n = new Node();
                    n.color = color;
                    //string[] s2 = s[i].Split(" ");
                    Move m = moves[i];

                    // 自玉がDiscovered Checkになってしまった場合
                    /*if (IsAttacked(BTree, BTree.SQ_King[color], color) > 0)
                    {
                        //UnDo(ref BTree, m, color);
                        continue;
                    }*/

                    int ifrom = m.From;
                    int ito = m.To;
                    int icap_pc = (int)m.CapPiece;

                    if (ifrom < Square_NB)
                    {
                        // PIN駒を動かしてしまった場合
                        Direction idirec = Adirec[ifrom, ito];
                        if (IsPinnedOnKing(BTree, ifrom, idirec, color) > 0)
                        {
                            //UnDo(ref BTree, m, color);
                            continue;
                        }
                    }
                    else
                    {
                        // 駒打ちなのに駒取りになってしまった場合 = > 多分発生しないが一応入れておく
                        if (icap_pc != (int)Piece.Empty)
                        {
                            //UnDo(ref BTree, m, color);
                            continue;
                        }
                    }

                    // 玉を取ってしまった場合
                    if (icap_pc == (int)Piece.King)
                    {
                        //UnDo(ref BTree, m, color);
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
                    if (i == 0)
                        start_index = n.ThisIndex;
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
                    int v = 0;
                    try
                    {
                        v = -(int)EvalWrapper(bt, color, bt.ply, m, false);
                        n.Value = v;
                    }
                    catch { v = Value_Min; }
                    //int result = PlayOutNoSearch(ref bt, color ^ 1, ply + 1, ref moves);
                    //int v = -EvalWrapper(bt, color, bt.ply, m, false);
                    n.Value = v;
                    List<Move> playout_moves = new List<Move>();
                    double result = PlayOutShallow(ref bt, color ^ 1, ply + 1, ref playout_moves, v);
                    //n.PolicyResult = Sigmoid((float)n.Value / 600);
                    NodeList.Add(n);
                    legal_moves.Add(m);
                    nodes_added++;
                    EvalNode(current_index, color ^ 1, v);
                    UpdateParam(current_index, result);

                    UnDo(ref BTree, m, color);

                    flag = true;
                }

                int[] colors = new int[nodes_added];
                Move[] prev_moves = new Move[nodes_added];
                for (int i = 0; i < nodes_added; i++)
                {
                    colors[i] = color;
                    prev_moves[i] = NodeList[parent_index].move;
                }

                List<double[]> li_ft = MakeInputFeatures(BTree, colors, legal_moves, prev_moves);
                torch.Tensor x = torch.zeros(nodes_added, input_num, dtype: torch.float32, device: d);
                var ar_ft = li_ft.ToArray();
                for (int j = 0; j < ar_ft.Length; j++)
                    x[j] = torch.tensor(ar_ft[j]);
                var y = seq.forward(x);
                List<double> li_p = new List<double>();
                int index = start_index;
                for (int j = 0; j < nodes_added; j++)
                {
                    double dv = y[j].ToDouble();
                    li_p.Add(dv);
                    NodeList[index++].PolicyResult = (float)dv;
                }

                //tt.PolicyMoves.TryAdd(BTree.CurrentHash, moves);
            }

            if (flag == true)
                NodeList[parent_index].IsLeaf = false;
        }


        private void EvalNode(int node_index, int color, int param_v)
        {
            if (tt.value.TryGetValue(BTree.CurrentHash, out int v))
            {
                // 勝率に変換する
                // 以下の式、元々は山本一成さん考案のはず。
                float f = Sigmoid(v / 600.0f);

                // トランスポジションテーブルにデータがあった場合
                NodeList[node_index].WinRateSum = f;
                NodeList[node_index].LostRateSum = 1 - f;
                NodeList[node_index].EvalCount++;
                return;
            }

            if (node_index == 7)
            {
                int xxxxxxx = 0;
            }

            //v = EvalWrapper(BTree, color, BTree.ply, NodeList[node_index].move, false);

            float v2 = Sigmoid((float)param_v / (float)600);

            // 手番側の勝率で学習させてある。1手指した後の手番の勝率が返ってくるので反転する。
            //NodeList[node_index].WinRateSum = 1 - v;
            //NodeList[node_index].LostRateSum = v;
            //NodeList[node_index].EvalCount++;// ToDo:要るかどうか分からないので要精査
            NodeList[node_index].WinRateSum = v2;
            NodeList[node_index].LostRateSum = 1 - v2;
            NodeList[node_index].EvalCount++;// ToDo:要るかどうか分からないので要精査

            // 評価値に変換する。
            // 【参考文献】
            // 山岡忠夫『将棋AIで学ぶディープラーニング』（マイナビ、2018年）P.222

            // トランスポジションテーブルに保存する
            tt.value.TryAdd(BTree.CurrentHash, param_v);
        }
        private void UpdateParam(int node_index, double result)
        {
            NodeList[node_index].TrialCount += 1;
            NodeList[node_index].RewardSum += (float)result;

            //if (node_index == 7)
            //Console.WriteLine("reward_sum=" + NodeList[node_index].RewardSum.ToString());
            //if (node_index == 7 && NodeList[node_index].RewardSum > 50)
            //{
            //int aaaaaaa = 0;
            //}
            /*if (result == 0)
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
            }*/
            Node current_node = NodeList[node_index];
            float delta;
            float delta2;
            float delta3;
            // 手番側の勝率で学習させてある
            if (current_node.color == (int)BTree.RootColor)
            {
                delta = NodeList[node_index].WinRateSum;
                delta2 = NodeList[node_index].LostRateSum;
                delta3 = NodeList[node_index].RewardSum;
            }
            else
            {
                delta = NodeList[node_index].LostRateSum;
                delta2 = NodeList[node_index].WinRateSum;
                delta3 = -NodeList[node_index].RewardSum;
            }
            if (current_node.ParentIndex == 0)
                return;
            while (true)
            {
                int index = current_node.ParentIndex;
                current_node = NodeList[index];
                NodeList[index].TrialCount += 1;
                NodeList[index].PlayoutCount += 1;
                /*if (result == 0)
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
                }*/

                // ※ ここは精査する必要がある。
                if (current_node.color == (int)BTree.RootColor)
                {
                    NodeList[index].WinRateSum += delta;
                    NodeList[index].LostRateSum += delta2;
                    NodeList[index].RewardSum += delta3;
                }
                else
                {
                    NodeList[index].WinRateSum += delta2;
                    NodeList[index].LostRateSum += delta;
                    NodeList[index].RewardSum -= delta3;
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
                    float u = NodeList[idx].PolicyResult * (float)Math.Sqrt(t) / (NodeList[idx].PlayoutCount + 1);
                    float q = 0.0F;
                    if (NodeList[idx].EvalCount > 0 && NodeList[idx].PlayoutCount > 0)
                        q = (float)(NodeList[i].WinRateSum / NodeList[i].EvalCount);
                    //q = (float)((1 - value_lambda) * (NodeList[idx].WinRateSum / NodeList[idx].EvalCount) + value_lambda * (NodeList[idx].WinCount / NodeList[idx].PlayoutCount));
                    ucb1_array[i++] = u + q;
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
                        bt = DeepCopy(bt, false);
                        List<Move> moves = new List<Move>();
                        int v = NodeList[max_index].Value;
                        float result = PlayOutShallow(ref bt, color ^ 1, ply + 1, ref moves, v);
                        NodeList[idx].PlayoutCount++;
                        PlayOutCount++;
                        EvalNode(idx, color ^ 1, v);
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
                    Node current_node = NodeList[idx];
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
            /*if (result == 0)
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
            }*/
            Node current_node = NodeList[node_index];
            float delta;
            float delta2;
            // 手番側の勝率で学習させてある
            if (current_node.color == (int)BTree.RootColor)
            {
                delta = NodeList[node_index].WinRateSum;
                delta2 = NodeList[node_index].LostRateSum;
            }
            else
            {
                delta = NodeList[node_index].LostRateSum;
                delta2 = NodeList[node_index].WinRateSum;
            }
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
                /*if (result == 0)
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
                }*/

                if (current_node.color == (int)BTree.RootColor)
                {
                    NodeList[index].WinRateSum += delta;
                    NodeList[index].LostRateSum += delta2;
                }
                else
                {
                    NodeList[index].WinRateSum += delta2;
                    NodeList[index].LostRateSum += delta;
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
