using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Antares.Analyze;
using static Antares.AttacksOperation;
using static Antares.Board;
using static Antares.Common;
using static Antares.GenMoves;
using static Antares.Mate1Ply;
using static Antares.Playout;
using static Antares.TrainReinforce3.Learn;
using static System.Net.Mime.MediaTypeNames;
using static TorchSharp.torch.distributions.constraints;
using static TorchSharp.torch.optim.lr_scheduler.impl.CyclicLR;
using BitBoard = System.UInt128;

namespace Antares
{
    // 深層学習型のモンテカルロ木探索クラス
    internal class MCTS4
    {
        public BoardTree BTree;
        public List<NodeDeep> NodeList = new List<NodeDeep>();
        public int PlayOutCount;
        public float[] RootOutput;
        public const float value_lambda = 0.3f;
        public const int nthr = 30;
        public int TaskNumber;
        public long SearchTimeLimit;// ミリ秒で指定する
        public const long TimeBuffer = 500;
        public Stopwatch sw = new Stopwatch();
        public Queue<string> queue_from_main_thread_p = new Queue<string>();
        public Queue<string> queue_to_main_thread_p = new Queue<string>();
        public Queue<string> queue_from_main_thread_v = new Queue<string>();
        public Queue<string> queue_to_main_thread_v = new Queue<string>();
        public TT2 tt = new TT2();
        public bool is_abort;
        public bool is_finished;

        public List<Move> TotalParam(int num_tasks, ref List<float> f, ref List<int> t, ref List<MCTS4> mcts_tree)
        {
            int value_max, max_index, limit;
            List<int> trial_count_array = new List<int>();
            List<float> win_rate_array = new List<float>();
            List<Move> return_moves = new List<Move>();
            List<float> return_win_rate_array = new List<float>();
            List<int> return_trial_count_array = new List<int>();
            MCTS4 temp_tree = new MCTS4();
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
                    for (int j = 0; j < BTree.RootMoves.Length; j++)
                    {
                        mcts_tree[0].NodeList[j + 1].TrialCount += temp_tree.NodeList[j + 1].TrialCount;
                        mcts_tree[0].NodeList[j + 1].WinCount += temp_tree.NodeList[j + 1].WinCount;
                        mcts_tree[0].NodeList[j + 1].DrawCount += temp_tree.NodeList[j + 1].DrawCount;
                        mcts_tree[0].NodeList[j + 1].LostCount += temp_tree.NodeList[j + 1].LostCount;
                        mcts_tree[0].NodeList[j + 1].EvalCount += temp_tree.NodeList[j + 1].EvalCount;
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
                    win_rate_array.Add((float)(mcts_tree[0].NodeList[i + 1].WinCount / (float)(trial_count_array[i])));
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
            NodeDeep best_move_node = mcts_tree[0].NodeList[max_node_index];
            NodeDeep current_node = new NodeDeep();
            NodeDeep node = new NodeDeep();
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

        public void SetRootOutput()
        {
            int[] newShape = new int[] { 1, 105, 9, 9 };
            int[] newShape2 = new int[] { 1, 32, 81 };
            float[] inputData = new float[1 * 105 * 9 * 9];
            Tensor<float> outputTensor = new DenseTensor<float>(new int[] { 1, 32, 9, 9 });
            inputData = MakeInputFeature(ref BTree, (int)BTree.RootColor);
            string inputName = sessions.policy_session.InputMetadata.Keys.First();
            DenseTensor<float> inputTensor = new DenseTensor<float>(inputData, newShape);
            // Create input container
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = sessions.policy_session.Run(inputs);
            outputTensor = results.First().AsTensor<float>();
            outputTensor = outputTensor.Reshape(newShape2);
            //float v_sum = 0.0f;
            List<float> li_v = new List<float>();
            for (int i = 0; i < BTree.RootMoves.Length; i++)
            {
                string s = CSA.Move2CSA(BTree.RootMoves[i]);
                Direction d = new Direction();
                int ifrom = BTree.RootMoves[i].From;
                int ito = BTree.RootMoves[i].To;
                //int idirec = (int)Adirec[ifrom, ito];
                int idirec = 0;
                float[,] lbl = MakeOutputLabel(BTree.RootMoves[i], ref d, ref idirec);
                float v = outputTensor[0, idirec, ito];
                //v_sum += v;
                li_v.Add(v);
            }
            RootOutput = new float[BTree.RootMoves.Length];
            for (int i = 0; i < BTree.RootMoves.Length; i++)
            {
                RootOutput[i] = li_v[i];
            }

            List<int> idxes = new List<int>();
            int limit = 4;
            if (BTree.RootMoves.Length <= limit)
            {
                limit = BTree.RootMoves.Length;
            }
            List<float> outputs = new List<float>();
            for (int i = 0; i < limit; i++)
            {
                int max_index = Array.IndexOf(RootOutput, RootOutput.Max());
                idxes.Add(max_index);
                outputs.Add(RootOutput[idxes[i]]);
                RootOutput[max_index] = float.MinValue;
            }
            List<Move> moves = new List<Move>();
            for (int i = 0; i < idxes.Count; i++)
            {
                moves.Add(BTree.RootMoves[idxes[i]]);
            }
            BTree.RootMoves = new Move[moves.Count];
            RootOutput = new float[moves.Count];
            float outputs_sum = outputs.Sum();
            for (int i = 0; i < idxes.Count; i++)
            {
                BTree.RootMoves[i] = moves[i];
                RootOutput[i] = outputs[i] / outputs_sum;
            }
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

            List<Move> legal_moves = new List<Move>();

            for (int i = 0; i < move_list.Count; i++)
            {
                if (IsMoveValid(ref BTree, move_list[i], (int)BTree.RootColor))
                    legal_moves.Add(move_list[i]);
            }

            BTree.RootMoves = legal_moves.ToArray();
        }

        public void Root()
        {
            float[] result_array = new float[BTree.RootMoves.Length];

            //Console.WriteLine("root_0");

            NodeList.Clear();

            NodeDeep root_node = new NodeDeep();
            NodeList.Add(root_node);

            for (int i = 0; i < BTree.RootMoves.Length; i++)
            {
                NodeDeep n = new NodeDeep();
                n.color = (int)BTree.RootColor;
                Move m = BTree.RootMoves[i];
                n.ParentIndex = 0;
                n.ThisIndex = i + 1;
                n.move = m;
                NodeList[0].ChildIndexes.Add(i + 1);
                n.PolicyResult = RootOutput[i];
                result_array[i] = RootOutput[i];
                NodeList.Add(n);
            }

            while(true)
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
                    float u = NodeList[i].PolicyResult * (float)Math.Sqrt(t) / (NodeList[i].PlayoutCount + 1);
                    float q = 0.0F;
                    if (NodeList[i].EvalCount > 0 && NodeList[i].PlayoutCount > 0)
                        q = (float)((1 - value_lambda) * (NodeList[i].WinRateSum / NodeList[i].EvalCount) + value_lambda * (NodeList[i].WinCount / NodeList[i].PlayoutCount));
                    ucb1_array[i - 1] = u + q;
                    i++;
                }

                int max_index = Array.IndexOf(ucb1_array, ucb1_array.Max()) + 1;
                Do(ref BTree, BTree.RootMoves[max_index - 1], (int)BTree.RootColor);
                if (NodeList[max_index].IsLeaf)
                {
                    elapsed = sw.ElapsedMilliseconds;
                    if (SearchTimeLimit < (elapsed + TimeBuffer))
                        break;
                    if (is_abort)
                        break;

                    if (NodeList[max_index].TrialCount >= nthr)
                    {
                        ExpandNode(((int)BTree.RootColor ^ 1), max_index, BTree.ply + 1);
                        if (is_abort)
                            break;
                    }
                    else
                    {
                        BoardTree bt = new BoardTree();
                        BoardTreeAlloc(ref bt);
                        bt = DeepCopy(BTree, false);
                        List<Move> ms = new List<Move>();
                        int result = Playout.PlayOutNoSearch(ref bt, ((int)BTree.RootColor ^ 1), max_index, ref ms);
                        //int result = Playout.PlayOutUseRolloutPolicy(ref bt, ((int)BTree.RootColor ^ 1), max_index, BTree.RootMoves[max_index - 1]);
                        EvalNode(max_index, ((int)BTree.RootColor ^ 1));
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

        private void ExpandNode(int color, int parent_index, int ply)
        {
            // 1手詰めと宣言勝ちをチェックし、勝ちがあった場合は手番側の勝ちをNodeに設定してreturnする。
            Move mate_move = new Move();
            BitBoard ulret = IsAttacked(BTree, BTree.SQ_King[color], color);
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
                GenEvasion(BTree, color, ref evasion_moves);
                // 王手を逃れる手が生成できなかったら詰み
                if (evasion_moves.Count == 0)
                {
                    NodeList[parent_index].WinRateSum = 1.0F;
                    NodeList[parent_index].LostRateSum = 0.0F;
                    NodeList[parent_index].EvalCount += 1;
                    return;
                }
            }

            int current_index;
            bool flag = false;

            // メインスレッドにPolicy Networkのリクエストを投げる
            string str_throw = SFEN.ToSFEN(BTree, color);
            str_throw = "p," + str_throw;
            queue_to_main_thread_p.Enqueue(str_throw);

            // メインスレッドから結果を受信するまで待機する
            while (queue_from_main_thread_p.Count == 0) { Thread.Sleep(1); if (is_abort) { return; } }

            string str_receive = queue_from_main_thread_p.Dequeue(); //データのフォーマットは"7776FU 0.65,2226FU 0.25,5556FU 0.1"といった感じを想定

            queue_from_main_thread_p.Clear();

            // 詰んでいた場合
            if (str_receive == "mate")
            {
                NodeList[parent_index].WinRateSum = 1.0F;
                NodeList[parent_index].LostRateSum = 0.0F;
                NodeList[parent_index].EvalCount += 1;
                return;
            }

            string[] s = str_receive.Split(',');
            List<Move> moves = new List<Move>();

            for (int i = 0; i < s.Length - 1; i += 2)
            {
                NodeDeep n = new NodeDeep();
                n.color = color;
                string[] s2 = s[i].Split(" ");
                Move m = CSA.CSA2Move(BTree, s2[0]);
                int ifrom = m.From;
                int ito = m.To;
                if (ifrom < Square_NB)
                { 
                    // Discovered Checkの場合
                    if (IsPinnedOnKing(BTree, ifrom, Adirec[ifrom, ito], color) != 0)
                    {
                        continue;
                    }
                }
                // 玉を取る非合法手の場合
                if (m.CapPiece == Piece.King)
                {
                    continue;
                }
                // Discovered Checkの場合
                Do(ref BTree, m, color);
                if (IsAttacked(BTree, BTree.SQ_King[color], color) != 0)
                {
                    UnDo(ref BTree, m, color);
                    continue;
                }
                n.ThisIndex = NodeList.Count;
                n.ParentIndex = NodeList[parent_index].ThisIndex;
                n.move = m;
                moves.Add(m);
                NodeList[parent_index].ChildIndexes.Add(n.ThisIndex);
                n.PolicyResult = float.Parse(s2[1]); //already executed softmax function.
                NodeList.Add(n);
                current_index = n.ThisIndex;
                BoardTree bt = new BoardTree();
                BoardTreeAlloc(ref bt);
                bt = DeepCopy(BTree, false);
                List<Move> ms = new List<Move>();
                int result = Playout.PlayOutNoSearch(ref bt, color ^ 1, ply + 1, ref ms);
                //int result = Playout.PlayOutUseRolloutPolicy(ref bt, color ^ 1, ply + 1, m);
                EvalNode(current_index, color ^ 1);
                UpdateParam(current_index, result);
                UnDo(ref BTree, m, color);
                flag = true;
            }
            if (flag == true)
                NodeList[parent_index].IsLeaf = false;
        }

        private void EvalNode(int node_index, int color)
        {
            if (tt.value.TryGetValue(BTree.CurrentHash, out float f))
            {
                // トランスポジションテーブルにデータがあった場合
                NodeList[node_index].WinRateSum = 1 - f;
                NodeList[node_index].LostRateSum = f;
                NodeList[node_index].EvalCount++;
                return;
            }

            // メインスレッドにValue Networkのリクエストを投げる
            string str_throw = SFEN.ToSFEN(BTree, color);
            str_throw = "v," + str_throw;
            queue_to_main_thread_v.Enqueue(str_throw);

            // メインスレッドから結果を受信するまで待機する
            while (queue_from_main_thread_v.Count == 0) { Thread.Sleep(1); if (is_abort) { return; } }


            string str_receive = queue_from_main_thread_v.Dequeue(); //データのフォーマットは"0.65"といった感じを想定

            float v = float.Parse(str_receive);

            // 手番側の勝率で学習させてある。1手指した後の手番の勝率が返ってくるので反転する。
            NodeList[node_index].WinRateSum = 1 - v;
            NodeList[node_index].LostRateSum = v;
            NodeList[node_index].EvalCount++;

            // トランスポジションテーブルに保存する
            tt.value.TryAdd(BTree.CurrentHash, v);
        }

        private void UpdateParam(int node_index, int result)
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
            NodeDeep current_node = NodeList[node_index];
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
            }
        }
        void DescendNode(int color, int node_index, int nthr, int ply)
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
                        q = (float)((1 - value_lambda) * (NodeList[idx].WinRateSum / NodeList[idx].EvalCount) + value_lambda * (NodeList[idx].WinCount / NodeList[idx].PlayoutCount));
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
                        bt = DeepCopy(BTree, false);
                        List<Move> ms = new List<Move>();
                        int result = Playout.PlayOutNoSearch(ref bt, color ^ 1, idx, ref ms);
                        //int result = Playout.PlayOutUseRolloutPolicy(ref bt, color ^ 1, idx, NodeList[idx].move);
                        EvalNode(idx, color ^ 1);
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
                    NodeDeep current_node = NodeList[idx];
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

        void AscendNode(int color, int node_index, int result)// パラメータ plyは要らないか？
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
            NodeDeep current_node = NodeList[node_index];
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
