using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static Antares.AttacksOperation;
using static Antares.Board;
using static Antares.Common;
using static Antares.Evaluate;
using static Antares.GenMoves;
using static Antares.Mate1Ply;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Antares
{
    // 実現確率探索
    public class AlphaBeta4
    {
        public static TT ttt = new TT();
        public const int futility_margin = 768; // Futility Pruningのマージン値 768
        public static int[] limit_table = new int[Ply_Max + 1];
        public static int[] ext_table = new int[Moves_Max];
        public static int thread_num = 1;
        public static int rest_thread_num = 0;
        public static AlphaBetaTree[] tlp_abt;
        public static Task[] tasks;
        const int input_num = 1170;
        const int middle_num = 32;
        const int output_num = 1;
        const double lr = 0.0008;
        const double mt = 0.9;
        const double wd = 0.0001;
        const string model_file_name = "model_tp.pth";
        const string optimizer_file_name = "optimizer_tp.pth";
        const int console_out_threshold = 25;
        const int backup_threshold = 1000;
        public static double trans_prob_threshold = 0.001;
        bool[] flg_color = { false, true };
        public static Device d = torch.device(DeviceType.CPU);
        public static TorchSharp.Modules.Linear input_layer = Linear(input_num, middle_num, device: d);
        public static TorchSharp.Modules.BatchNorm1d input_norm = BatchNorm1d(middle_num, 2e-05, device: d);
        public static TorchSharp.Modules.Linear output_layer = Linear(middle_num, output_num, device: d);
        public static TorchSharp.Modules.Sequential seq = Sequential(("input_layer", input_layer), ("input_norm", input_norm), ("input_relu", ReLU()), ("output_layer", output_layer), ("output_sigmoid", Sigmoid()));


        public struct AlphaBetaTree
        {
            public Mate1Ply m1p;
            public GenMoves gm;
            //AttacksOperation atkop;
            //BitOperation bitop;
            public List<Move>[] moves;
            public BoardTree bt;
            public TT tt;
            public int task_number;
            public Move[] current_move;
            public int[] eval;
            public int[] eval_pp;// 玉を動かした時の差分計算に使う。
            public List<Move> pv;
            public Move[] pv2;
            public int pv_length;
            public uint size_tt;
            public Sort.MoveAndProb[,] map_before;
            public Sort.MoveAndProb[,] map_after;
            public Sort.MoveAndProb[,] map_quies_before;
            public Sort.MoveAndProb[,] map_quies_after;
            public int iteration_max;
            public int root_move_num;
            public bool[] is_check;
            public bool is_abort;
            public bool is_finished;
            public long num_node_searched;
            public long null_move_cut;
            public long delta_cut;
            public long razor_cut;
            public long futility_cut;
            public long hash_cut;
            public int BestValue;
            //public ValueTT vtt;

            public AlphaBetaTree()
            {
                m1p = new Mate1Ply();
                gm = new GenMoves();
                moves = new List<Move>[Ply_Max];
                for (int i = 0; i < Ply_Max; i++)
                {
                    moves[i] = new List<Move>();
                }
                bt = new BoardTree();
                tt = new TT();
                task_number = 0;
                current_move = new Move[Ply_Max];
                eval = new int[Ply_Max];
                eval_pp = new int[Ply_Max];
                pv = new List<Move>();
                pv_length = 0;
                size_tt = 0x10000000; // 256MB
                map_before = new Sort.MoveAndProb[Ply_Max, Moves_Max];
                map_after = new Sort.MoveAndProb[Ply_Max, Moves_Max];
                map_quies_before = new Sort.MoveAndProb[Ply_Max, CapMoves_Max];
                map_quies_after = new Sort.MoveAndProb[Ply_Max, CapMoves_Max];
                iteration_max = 0;
                root_move_num = 0;
                is_check = new bool[Ply_Max];
                is_abort = false;
                is_finished = false;
                num_node_searched = 0;
                null_move_cut = 0;
                delta_cut = 0;
                razor_cut = 0;
                futility_cut = 0;
                hash_cut = 0;
                BestValue = 0;
                //vtt = new ValueTT();
            }

            public enum NodeState : uint
            {
                node_pv = 1,
                node_do_null_move = 2,
                node_do_delta = 4,
                node_do_razoring = 8,
                node_do_futility = 16,
                node_do_probcut = 32,
                node_mate_threat = 128,
                node_slave = 256,
                node_learning = 512,
            };

            public int[] CapExTable = { 0, 1, 2, 2, 4, 4, 8, 8, 0, 2, 3, 3, 4, 0, 8, 8 };
        }

        public static void InitLimitTable()
        {
            for (int i = 0; i < Ply_Max; i++)
            {
                if (i <= 8)
                {
                    limit_table[i] = 4;
                }
                else if (i >= 9 && i <= 16)
                {
                    limit_table[i] = 6;
                }
                else if (i >= 17 && i <= 24)
                {
                    limit_table[i] = 8;
                }
                else if (i >= 25 && i <= 32)
                {
                    limit_table[i] = 10;
                }
                else if (i >= 33 && i <= 40)
                {
                    limit_table[i] = 12;
                }
                else if (i >= 41 && i <= 48)
                {
                    limit_table[i] = 14;
                    //limit_table[i] = 18;
                }
                else if (i >= 49 && i <= 56)
                {
                    limit_table[i] = 16;
                    //limit_table[i] = 22;
                }
                else if (i >= 57 && i <= 64)
                {
                    limit_table[i] = 18;
                    //limit_table[i] = 26;
                }
                else
                {
                    limit_table[i] = 20;
                    //limit_table[i] = 50;
                }
            }

            for (int i = 0; i < ext_table.Length; i++)
            {
                if (i >= 0 && i < 4)
                {
                    ext_table[i] += Ply_Inc / 2;
                }
                else if (i >= 4 && i < 8)
                {
                    ext_table[i] += Ply_Inc / 4;
                }
                else if (i >= 8 && i < 16)
                {
                    //extension += Ply_Inc / 8;

                }
                else if (i >= 16 && i < 18)
                {
                    ext_table[i] -= Ply_Inc / 4;
                }
                else if (i >= 18)
                {
                    ext_table[i] -= Ply_Inc / 2;
                }
            }
        }

        public static void SetThreadNum(int n_threads)
        {
            thread_num = n_threads;
            rest_thread_num = n_threads - 1;
        }

        public static void InitTlpAbt(int n_threads)
        {
            tlp_abt = new AlphaBetaTree[n_threads - 1];
            tasks = new Task[n_threads - 1];
            for (int i = 0; i < n_threads - 1; i++)
            {
                tlp_abt[i] = new AlphaBetaTree();
                tlp_abt[i].bt = new BoardTree();
                BoardTreeAlloc(ref tlp_abt[i].bt);
                tasks[i] = new Task(() => { });
            }
        }

        public static void LoadTransProbs()
        {
            seq.load(model_file_name);
            seq.eval();
        }

        public static void TlpSearchWrapper(ref AlphaBetaTree abt, int color, int alpha, int beta, int ply, uint state_node, double trans_prob, ref int return_value, ref List<Move> temp_pv)
        {
            return_value = -Search(ref abt, color, alpha, beta, ply, state_node, trans_prob, ref temp_pv);
        }

        public static int Search(ref AlphaBetaTree abt, int color, int alpha, int beta, int ply, uint state_node, double trans_prob, ref List<Move> temp_pv)
        {
            int value, i, j, k, v, iret, temp_value, move_count, ifrom, ito, icap_pc, alpha_old, thread_index, prev_pv_length;
            uint state_node_new;
            Direction idirec;
            abt.moves[ply].Clear();
            List<Move> moves = abt.moves[ply];
            List<Move> local_pv = new List<Move>();
            List<Move> tpv = new List<Move>();
            List<Move> tlp_tpv = new List<Move>();
            Move m = new Move();
            abt.num_node_searched++;
            value = 0;
            thread_index = 0;
            prev_pv_length = 0;

            if (!abt.is_check[ply - 1] && trans_prob < trans_prob_threshold)
            {
                value = (color == 0) ? Eval(abt.bt) : -Eval(abt.bt);
                abt.tt.Store(abt.bt.CurrentHash, value, (Common.Color)(color ^ 1), abt.is_check[ply - 1], abt.current_move[ply - 1], ply);
                return value;
            }

            if (!abt.is_check[ply - 1] && ply >= 64)
            {
                value = (color == 0) ? Eval(abt.bt) : -Eval(abt.bt);
                abt.tt.Store(abt.bt.CurrentHash, value, (Common.Color)(color ^ 1), abt.is_check[ply - 1], abt.current_move[ply - 1], ply);
                return value;
            }

            // ply > 1は取りあえず入れていない。
            // 前の手が王手だったら1手詰めは発動できない。
            if (!abt.is_check[ply - 1])
            {
                Move mate_move = Mate1Ply.MateIn1Ply(abt.bt, color);
                if (mate_move.Value != 0)
                {
                    value = Value_Mate;
                    abt.current_move[ply] = mate_move;
                    abt.tt.Store(abt.bt.CurrentHash, value, (Antares.Common.Color)color, true, mate_move, ply);
                    if (alpha < value && value < beta)
                    {
                        temp_pv.Add(mate_move);
                    }
                    abt.bt.EvalArray[ply] = value;
                    return value;
                }
            }

            // 宣言勝ちできるかどうかを調べる
            // 0:宣言勝ちの局面ではない→スルーする
            // 1:先手の勝ち
            // 2:後手の勝ち
            iret = IsDeclarationWin(abt.bt);
            switch (iret)
            {
                case 1:
                    if (color == 0)
                    {
                        value = Value_Max;
                        abt.tt.Store(abt.bt.CurrentHash, value, (Antares.Common.Color)color, false, m, ply);
                    }
                    else
                    {
                        value = Value_Min;
                        abt.tt.Store(abt.bt.CurrentHash, value, (Antares.Common.Color)color, false, m, ply);
                    }
                    break;
                case 2:
                    if (color == 1)
                    {
                        value = Value_Max;
                        abt.tt.Store(abt.bt.CurrentHash, value, (Antares.Common.Color)color, false, m, ply);
                    }
                    else
                    {
                        value = Value_Min;
                        abt.tt.Store(abt.bt.CurrentHash, value, (Antares.Common.Color)color, false, m, ply);
                    }
                    break;
            }
            if (iret != 0)
            {
                abt.bt.EvalArray[ply] = value;
                return value;
            }

            // 千日手かどうかを調べる
            // 0:千日手ではない→スルーする
            // 1:千日手
            // 2:連続王手の千日手
            iret = IsRepetition(abt.bt, abt.tt);
            switch (iret)
            {
                case 1:
                    value = Value_Draw;
                    abt.tt.Store(abt.bt.CurrentHash, value, (Antares.Common.Color)color, false, m, ply);
                    break;
                case 2:
                    value = Value_Max;
                    abt.tt.Store(abt.bt.CurrentHash, value, (Antares.Common.Color)color, true, m, ply);
                    break;
            }
            if (iret != 0)
            {
                abt.bt.EvalArray[ply] = value;
                return value;
            }

            alpha_old = alpha;

            // #トランスポジションテーブルを調べる
            if (abt.tt.value.ContainsKey(abt.bt.CurrentHash) && abt.tt.color.ContainsKey(abt.bt.CurrentHash) && abt.tt.ply.ContainsKey(abt.bt.CurrentHash))
            {
                int c = (int)abt.tt.color[abt.bt.CurrentHash];
                int temp_ply = abt.tt.ply[abt.bt.CurrentHash];
                if (c == color && ply > temp_ply)// 前回plyより大きくないと、探索なしで終わってしまう。
                {
                    // ハッシュ値と手番が一致した場合
                    abt.hash_cut++;
                    value = abt.tt.value[abt.bt.CurrentHash];
                    abt.bt.EvalArray[ply] = value;
                    return value;
                }
            }


            // Null Move Pruning => ※仮置きで外してあるが、多分有効である。
            /*if (!abt.is_check[ply - 1] && 2 * Ply_Inc <= depth && (state_node & (uint)AlphaBetaTree.NodeState.node_do_null_move) != 0 && beta <= temp_value)
            {
                // PVノードではないので枝刈りは実行しない。
                state_node_new = 0;
                m.SetNullMove();
                abt.current_move[ply] = m;
                abt.is_check[ply] = false; // Null Moveの時は王手フラグを立てない。
                //abt.prev_eval[ply] = -temp_value;
                DoNull(ref abt.bt);
                int null_value = -Search(ref abt, color ^ 1, -beta, 1 - beta, 1 * Ply_Inc, ply + 1, state_node_new, -prev_value);
                UnDoNull(ref abt.bt);
                m = new Move();
                abt.current_move[ply] = m;
                if (beta <= null_value)
                {
                    abt.null_move_cut++;
                    Move null_move = new Move();
                    null_move.SetNullMove();
                    abt.tt.Store(abt.bt.CurrentHash, null_value, (Common.Color)color, false, null_move, ply);
                    return null_value;
                }

                if (null_value == Value_Min)
                {
                    state_node |= (uint)AlphaBetaTree.NodeState.node_mate_threat;
                }
            }*/

            // Delta Pruningを実行する。
            // 【参考】
            // https://www.chessprogramming.org/Delta_Pruning
            // 1536は1024 + 512とか2のn乗の値を2つ足しただけで、科学的な根拠はない。
            /*if (!abt.is_check[ply - 1] && 2 * Ply_Inc <= depth && (state_node & (uint)AlphaBetaTree.NodeState.node_do_delta) != 0)
            {
                if (temp_value >= beta + 1536)
                {
                    abt.delta_cut++;
                    return beta;
                }

                if (temp_value < alpha - 1536)
                {
                    abt.delta_cut++;
                    return alpha;
                }

                if (alpha < temp_value)
                    alpha = temp_value;
            }*/

            // Razoringを実行する。直前の手が王手だったら実行しない。
            // Stockfishとは少し条件を変えてある。
            // ※これを入れる場合はライセンスがGPL3になる。=> ライセンスはMIT
            // 【参考】
            // https://www.chessprogramming.org/Razoring
            /*if (!abt.is_check[ply - 1] && 2 * Ply_Inc <= depth && temp_value < (alpha - 512 - 256 * (depth / Ply_Inc) * (depth / Ply_Inc)) && (state_node & (uint)AlphaBetaTree.NodeState.node_do_null_move) != 0)
            {
                // 局面更新をしていないので、手番そのままで反転なし。
                int rv = QuiesSearch(ref abt, color, alpha - 1, alpha, ply, 1, temp_value);
                if (rv < alpha && Math.Abs(rv) < Value_Min)
                {
                    abt.razor_cut++;
                    return rv;
                }
            }*/

            // 手番のみ相手に渡した場合に自玉に1手詰めがあるかを確認する。
            Move threat_move = Mate1Ply.MateIn1Ply(abt.bt, color ^ 1);

            // 手を生成する
            move_count = 0;
            if (!abt.is_check[ply - 1])
            {
                if (threat_move.Value == 0)
                {
                    GenDrop(abt.bt, color, ref moves);
                    GenNoCap(abt.bt, color, ref moves);
                    GenCap(abt.bt, color, ref moves);
                }
                else
                {
                    // 自玉に1手詰めがある場合
                    GenCheck(abt.bt, color, ref moves);
                    GenCapTreatPiece(abt.bt, color, ref moves, threat_move);
                    GenKingMove(abt.bt, color, ref moves);
                    GenInterfere(abt.bt, color, ref moves, threat_move);
                }
            }
            else
            {
                GenEvasion(abt.bt, color, ref moves);
            }

            move_count = moves.Count;

            if (move_count == 0)
            {
                value = Value_Min;
                abt.tt.Store(abt.bt.CurrentHash, value, (Antares.Common.Color)color, false, m, ply);
                abt.eval[ply] = value;
                return value;
            }

            torch.Tensor x = torch.zeros(move_count, input_num, dtype: torch.float32, device: d);
            int[] colors = new int[move_count];
            Move[] prev_moves = new Move[move_count];
            for (k = 0; k < move_count; k++)
                colors[k] = color;
            
            if (abt.bt.ply == 1)
            {
                for (k = 0; k < move_count; k++)
                    prev_moves[k] = new Move();
            }
            else
            {
                for (k = 0; k < move_count; k++)
                    prev_moves[k] = abt.current_move[ply - 1];
            }

            List<double[]> li_ft = TransProbs6.MakeInputFeatures(abt.bt, colors, moves, prev_moves);
            var ar_ft = li_ft.ToArray();
            for (k = 0; k < ar_ft.Length; k++)
                x[k] = torch.tensor(ar_ft[k]);

            double[] trans_probs = new double[move_count];
            var y = seq.forward(x);
            for (k = 0; k < ar_ft.Length; k++)
                trans_probs[k] = y[k].ToDouble();

            for (i = 0; i < move_count; i++)
            {
                Do(ref abt.bt, moves[i], color);

                // 自玉がDiscovered Checkになってしまった場合
                if (IsAttacked(abt.bt, abt.bt.SQ_King[color], color) > 0)
                    trans_probs[i] = 0.0;

                Sort.MoveAndProb temp_map = new Sort.MoveAndProb();
                temp_map.move = moves[i];
                temp_map.trans_prob = trans_probs[i];
                abt.map_before[ply, i] = temp_map;
                UnDo(ref abt.bt, moves[i], color);
            }

            if (move_count > 1)
            {
                Sort.MergeSort(ref abt.map_before, ref abt.map_after, 0, move_count, ply);
            }
            else
            {
                abt.map_after[ply, 0] = abt.map_before[ply, 0];
            }

            bool tlp_flag = false;
            int best_value = Value_Min;
            int first_value = best_value;

            int limit = limit_table[ply];// ※ ある程度有効だが、やや枝刈り過ぎの感じ => 少しWindowを広げてFutility Pruningと組み合わせるべきか？
            //limit = 20;

            // ※実現確率だけだと、枝を刈ることができないことが多いので、Futility Pruningもどきの条件を入れてみる。
            for (i = 0; i < move_count; i++)
            {
                int return_value = 0;

                // move count base pruning => これは枝を刈りすぎていた。
                /*if (i >= limit)
                {
                    break;
                }*/

                if (i == move_count - 1 || i == limit)
                {
                    //abt.futility_cut++;
                    if (tlp_flag == true)
                    {
                        while (!tasks[thread_index].IsCompleted) { }
                        if ((state_node & (uint)AlphaBetaTree.NodeState.node_pv) > 0 && (state_node & (uint)AlphaBetaTree.NodeState.node_learning) == 0 && (state_node & (uint)AlphaBetaTree.NodeState.node_slave) == 0 && tasks[thread_index].IsCompleted && tlp_flag == true)
                        {
                            if (return_value > best_value && (((temp_pv.Count + tlp_tpv.Count) >= prev_pv_length && tlp_tpv.Count >= tpv.Count) || return_value == Value_Mate))// && ply > tlp_abt[thread_index].pv_length
                            {
                                alpha = Math.Max(alpha, return_value);

                                {
                                    value = return_value;
                                    if (value > best_value)// ※ここは多分おかしい。
                                    {
                                        best_value = value;
                                        if (i == 0)
                                        {
                                            best_value = value;
                                            for (j = 0; j < tlp_tpv.Count; j++)
                                            {
                                                local_pv.Insert(0, tlp_tpv[j]);
                                            }
                                            local_pv.Insert(0, tlp_abt[thread_index].current_move[ply]);
                                        }
                                        else
                                        {
                                            best_value = value;
                                            local_pv.Clear();
                                            for (j = 0; j < tlp_tpv.Count; j++)
                                            {
                                                local_pv.Insert(0, tlp_tpv[j]);
                                            }
                                            local_pv.Insert(0, tlp_abt[thread_index].current_move[ply]);
                                        }
                                        abt.bt.EvalArray[ply] = alpha;
                                        abt.tt.Store(abt.bt.CurrentHash, value, (Common.Color)color, false, tlp_abt[thread_index].current_move[ply], ply);
                                        //abt.pv2[ply] = tlp_abt[thread_index].current_move[ply];
                                    }
                                }
                            }
                            rest_thread_num++;
                            tlp_flag = false;

                            if (ply == 1)
                                break;

                            // βカット
                            if (alpha >= beta)
                                return alpha;
                        }
                    }
                }

                if (i == limit)
                {
                    abt.futility_cut++;
                    break;
                }

                //int current_score = -abt.mas_after[ply, i].score;


                // Futility Pruningもどき
                //if (i > 0 && score_diff >= 768)
                /*int score_diff = score_first - current_score;
                if (i > 0 && score_diff >= futility_margin)
                {
                    abt.futility_cut++;
                    break;
                }*/

                /*int score_diff_prev = prev_value - current_score;

                if (i > 0 && score_diff_prev >= futility_margin)
                {
                    abt.futility_cut++;
                    break;
                }*/

                //abt.bt.EvalArray[ply] = -abt.mas_after[ply, i].score;

                Move current_move = abt.map_after[ply, move_count - i - 1].move;
                abt.current_move[ply] = current_move;
                double current_trans_prob = trans_prob * abt.map_after[ply, move_count - i - 1].trans_prob;

                // ToDo: 駒損を放置するPVが返ってくることが多いので、前の手とTo位置が同じ場合は0.25手か0.5手延長するのがベターか？

                // 1手進める。
                Do(ref abt.bt, current_move, color);

                // 自玉がDiscovered Checkになってしまった場合
                if (IsAttacked(abt.bt, abt.bt.SQ_King[color], color) > 0)
                {
                    //limit++; // 探索する手を1つ増やす。
                    UnDo(ref abt.bt, current_move, color);
                    continue;
                }

                abt.is_check[ply] = false;
                ifrom = current_move.From;
                ito = current_move.To;
                icap_pc = (int)current_move.CapPiece;

                if (ifrom < Square_NB)
                {
                    // PIN駒を動かしてしまった場合
                    idirec = Adirec[ifrom, ito];
                    if (IsPinnedOnKing(abt.bt, ifrom, idirec, color) > 0)
                    {
                        //limit++; // 探索する手を1つ増やす。
                        UnDo(ref abt.bt, current_move, color);
                        continue;
                    }
                }
                else
                {
                    // 駒打ちなのに駒取りになってしまった場合 = > 多分発生しないが一応入れておく
                    if (icap_pc != (int)Piece.Empty)
                    {
                        UnDo(ref abt.bt, current_move, color);
                        continue;
                    }
                }

                // 玉を取ってしまった場合
                if (icap_pc == (int)Piece.King)
                {
                    UnDo(ref abt.bt, current_move, color);
                    continue;
                }


                // 指した手が王手だったら王手フラグを立てる。
                if (IsAttacked(abt.bt, abt.bt.SQ_King[color ^ 1], color ^ 1) > 0)
                {
                    abt.is_check[ply] = true;
                }

                if ((state_node & (uint)AlphaBetaTree.NodeState.node_pv) > 0)
                {
                    state_node_new = (uint)AlphaBetaTree.NodeState.node_pv | (uint)AlphaBetaTree.NodeState.node_do_null_move
                        | (uint)AlphaBetaTree.NodeState.node_do_delta | (uint)AlphaBetaTree.NodeState.node_do_razoring
                        | (uint)AlphaBetaTree.NodeState.node_do_futility | (uint)AlphaBetaTree.NodeState.node_do_probcut;
                }
                else
                {
                    state_node_new = 0;
                }

                // ※ 同一ノードでのマルチタスク探索は1回だけにした方が効率が良いと思われる。
                if (i > 0 && (state_node & (uint)AlphaBetaTree.NodeState.node_pv) > 0 && (state_node & (uint)AlphaBetaTree.NodeState.node_learning) == 0 && (state_node & (uint)AlphaBetaTree.NodeState.node_slave) == 0 && rest_thread_num > 0 && current_trans_prob > (trans_prob_threshold * 1.3) &&  tlp_flag == false)
                {
                    // マルチスレッドで探索する。
                    rest_thread_num--;
                    tlp_flag = true;
                    thread_index = thread_num - 2 - rest_thread_num;
                    tlp_abt[thread_index].bt = DeepCopy(abt.bt, false);
                    tlp_abt[thread_index].tt = abt.tt;
                    int l = ply;
                    for (int n = 0; n <= l; n++)
                    {
                        tlp_abt[thread_index].current_move[n] = abt.current_move[n];
                        tlp_abt[thread_index].is_check[n] = abt.is_check[n];
                        tlp_abt[thread_index].bt.EvalArray[n] = abt.bt.EvalArray[n];
                    }
                    tlp_tpv.Clear();
                    tasks[thread_index] = Task.Run(() => TlpSearchWrapper(ref tlp_abt[thread_index], color ^ 1, -beta, -alpha,  ply + 1, (state_node_new | (uint)AlphaBetaTree.NodeState.node_slave), current_trans_prob, ref return_value, ref tlp_tpv));
                    UnDo(ref abt.bt, current_move, color);
                    continue;
                }

                // 次のノードを展開する。
                tpv.Clear();// 多分これが必要
                value = -Search(ref abt, color ^ 1, -beta, -alpha,  ply + 1, state_node_new, current_trans_prob, ref tpv);// ※残り深さとstate_nodeは後で変更する。

                // 1手戻す。
                UnDo(ref abt.bt, current_move, color);

                alpha = Math.Max(alpha, value);

                if (i == 0)
                {
                    best_value = value;
                    first_value = best_value;
                }
                else if (i > 0 && value > best_value)
                {
                    best_value = value;
                }


                // βカット
                if (alpha >= beta)
                {
                    break;
                }

                if (alpha != alpha_old)
                {
                    if (i == 0)
                    {
                        best_value = value;
                        for (j = 0; j < tpv.Count; j++)
                        {
                            local_pv.Insert(0, tpv[j]);
                        }
                        local_pv.Insert(0, current_move);
                        prev_pv_length = temp_pv.Count + tpv.Count + 1;
                        abt.bt.EvalArray[ply] = alpha;
                        abt.tt.Store(abt.bt.CurrentHash, value, (Common.Color)color, false, abt.current_move[ply], ply);
                    }
                    else
                    {
                        if (ply != 1)
                        {
                            if ((temp_pv.Count + tpv.Count) >= prev_pv_length || return_value == Value_Mate)
                            {
                                best_value = value;
                                local_pv.Clear();
                                for (j = 0; j < tpv.Count; j++)
                                {
                                    local_pv.Insert(0, tpv[j]);
                                }
                                local_pv.Insert(0, current_move);
                                prev_pv_length = temp_pv.Count + tpv.Count + 1;
                                abt.bt.EvalArray[ply] = alpha;
                                abt.tt.Store(abt.bt.CurrentHash, value, (Common.Color)color, false, abt.current_move[ply], ply);
                            }
                            else
                            {
                                alpha = alpha_old;
                            }
                        }
                        else
                        {
                            if ((temp_pv.Count + tpv.Count) >= prev_pv_length || return_value == Value_Mate)
                            {
                                best_value = value;
                                local_pv.Clear();
                                for (j = 0; j < tpv.Count; j++)
                                {
                                    local_pv.Insert(0, tpv[j]);
                                }
                                local_pv.Insert(0, current_move);
                                prev_pv_length = temp_pv.Count + tpv.Count + 1;
                                abt.bt.EvalArray[ply] = alpha;
                                abt.tt.Store(abt.bt.CurrentHash, value, (Common.Color)color, false, abt.current_move[ply], ply);
                            }
                            else
                            {
                                alpha = alpha_old;
                            }
                        }
                    }
                }

                alpha_old = alpha;
            }

            if (ply > 1)
            {
                for (i = 0; i < local_pv.Count; i++)
                {
                    temp_pv.Insert(0, local_pv[i]);
                }

            }
            else
            {
                for (i = 0; i < local_pv.Count; i++)
                {
                    temp_pv.Add(local_pv[i]);
                }
            }

            return alpha;
        }

        // ※静止探索は現状発動させていない。

        // 静止探索
        // 王手がかかっている場合は呼んではいけない。
        // ToDo: 以下の条件で手をソートする。
        // (1) 最も価値の高い駒を取る手を1番目にする。
        // (2) 直前の手の取り返しを2番目にする。
        // (3) 以下は価値の高い駒を取る手を上にする。 => ただし、と金、成香、成桂あたりは考慮が必要。
        // ※ 指した手はcurrent_moveに入っているので、判定はできるはず。
        /*public static int QuiesSearch(ref AlphaBetaTree abt, int color, int alpha, int beta, int ply, int qui_ply, int prev_value)
        {
            int value, alpha_old, stand_pat, move_count, i, j;
            int ifrom, ito, icap_pc;
            Direction idirec;
            abt.moves[ply].Clear();
            List<Move> moves = abt.moves[ply];
            alpha_old = alpha;

            stand_pat = prev_value;

            if (alpha < stand_pat)
            {
                if (beta <= stand_pat)
                {
                    Move m = new Move();
                    abt.current_move[ply] = m;// 現状 m = 0だが、BonanzaだとPassになっている。
                    return stand_pat;
                }
                alpha = stand_pat;
            }

            if (ply >= Ply_Max - 1)
            {
                Move m = new Move();
                // Bonanzaだとpvをcloseしている。
                abt.current_move[ply] = m;// 現状 m = 0だが、BonanzaだとMove_NAになっている。
                abt.tt.Store(abt.bt.CurrentHash, stand_pat, (Common.Color)(color ^ 1), abt.is_check[ply - 1], abt.current_move[ply - 1], ply);
                return stand_pat;
            }

            if (!abt.is_check[ply - 1])
            {
                Move mate_move = Mate1Ply.MateIn1Ply(abt.bt, color);
                if (mate_move.Value != 0)
                {
                    value = Value_Max;
                    abt.current_move[ply] = mate_move;
                    abt.tt.Store(abt.bt.CurrentHash, value, (Antares.Common.Color)color, true, mate_move, ply);
                    if (alpha < value && value < beta)
                    {
                        for (i = 0; i < ply; i++)
                        {
                            abt.pv[i] = abt.current_move[i];
                        }
                        abt.pv_length = ply;
                        abt.pv[ply] = mate_move;
                    }
                    abt.eval[ply] = value;
                    return value;
                }
            }

\

            if (!abt.is_check[ply - 1] && ply >= 64)
            {
                value = EvalWrapper(abt.bt, color ^ 1, ply, abt.current_move[ply - 1], false);
                abt.tt.Store(abt.bt.CurrentHash, value, (Common.Color)(color ^ 1), abt.is_check[ply - 1], abt.current_move[ply - 1], ply);
                return value;
            }

            move_count = 0;
            GenCap(abt.bt, color, ref moves);

            move_count = moves.Count;

            if (move_count == 0)
            {
                //value = Value_Min;
                //abt.tt.Store(abt.bt.CurrentHash, value, (Antares.Common.Color)color, false, m, ply);
                //abt.eval[ply] = value;
                return alpha;
            }

            int v;
            for (i = 0; i < move_count; i++)
            {
                Do(ref abt.bt, moves[i], color);
                //int v = (color == 0) ? EvalWrapper(abt.bt, color, moves[i], false) : -EvalWrapper(abt.bt, color, moves[i], false);
                // 自玉がDiscovered Checkになってしまった場合
                if (IsAttacked(abt.bt, abt.bt.SQ_King[color], color) > 0)
                {
                    v = Value_Min;
                }
                else
                {
                    v = EvalWrapper(abt.bt, color, ply, moves[i], false);
                }
                Sort.MoveAndScore temp_mas = new Sort.MoveAndScore();
                temp_mas.move = moves[i];
                temp_mas.score = v;
                abt.mas_quies_before[ply, i] = temp_mas;
                UnDo(ref abt.bt, moves[i], color);
            }

            if (move_count > 1)
            {
                Sort.MergeSort(ref abt.mas_quies_before, ref abt.mas_quies_after, 0, move_count, ply);
            }
            else
            {
                abt.mas_quies_after[ply, 0] = abt.mas_quies_before[ply, 0];
            }

            int score_first = -abt.mas_quies_after[ply, 0].score;
            Move prev_move = abt.current_move[ply - 1];

            for (i = 0; i < move_count; i++)
            {

                int current_score = -abt.mas_quies_after[ply, i].score;

                int score_diff = score_first - current_score;
                if (i > 0 && score_diff >= futility_margin)
                {
                    abt.futility_cut++;
                    break;
                }

                // ※ここはまだ精査していない。
                int score_diff_prev = prev_value - current_score;

                if (i > 0 && score_diff_prev >= futility_margin)
                {
                    abt.futility_cut++;
                    break;
                }

                abt.bt.EvalArray[ply] = -abt.mas_quies_after[ply, i].score;

                Move current_move = moves[i];
                abt.current_move[ply] = moves[i];

                // 取り合いの手かつ歩以外の駒を取る手のみ読む
                if (current_move.To != prev_move.To && current_move.CapPiece < Piece.Lance)
                    continue;

                // 1手進める。
                Do(ref abt.bt, current_move, color);

                // 自玉がDiscovered Checkになってしまった場合
                if (IsAttacked(abt.bt, abt.bt.SQ_King[color], color) > 0)
                {
                    //limit++; // 探索する手を1つ増やす。
                    UnDo(ref abt.bt, current_move, color);
                    continue;
                }

                abt.is_check[ply] = false;
                ifrom = current_move.From;
                ito = current_move.To;
                icap_pc = (int)current_move.CapPiece;

                if (ifrom < Square_NB)
                {
                    // PIN駒を動かしてしまった場合
                    idirec = Adirec[ifrom, ito];
                    if (IsPinnedOnKing(abt.bt, ifrom, idirec, color) > 0)
                    {
                        //limit++; // 探索する手を1つ増やす。
                        UnDo(ref abt.bt, current_move, color);
                        continue;
                    }
                }

                // 玉を取ってしまった場合
                if (icap_pc == (int)Piece.King)
                {
                    UnDo(ref abt.bt, current_move, color);
                    continue;
                }

                // 指した手が王手だったら王手フラグを立てる。
                if (IsAttacked(abt.bt, abt.bt.SQ_King[color ^ 1], color ^ 1) > 0)
                    abt.is_check[ply] = true;

                value = -QuiesSearch(ref abt, color ^ 1, -beta, -alpha, ply + 1, qui_ply + 1, -abt.mas_quies_after[ply, i].score);

                // 1手戻す。
                UnDo(ref abt.bt, current_move, color);

                if (alpha > alpha_old && alpha > abt.eval[ply] && ply > abt.pv_length)
                //if (alpha != alpha_old && ply > abt.pv_length)
                {
                    v = value;
                    if (color != (int)abt.bt.RootColor)
                        v = -v;
                    //if (depth <= Ply_Inc)// 後で変えるかもしれない。
                    if (v > abt.BestValue)// ※ここは多分おかしい。
                    {
                        abt.BestValue = v;
                        abt.pv.Clear();
                        for (j = 1; j < ply; j++)
                            abt.pv.Add(abt.current_move[j]);
                        abt.pv_length = abt.pv.Count;
                        if (value == Value_Min && abt.current_move[ply].Value != 0)
                        {
                            abt.pv.Add(abt.current_move[ply]);
                            abt.pv_length += 1;
                        }
                        abt.eval[ply] = alpha;
                        abt.tt.Store(abt.bt.CurrentHash, value, (Common.Color)color, false, abt.current_move[ply], ply);
                    }
                }
            }
            return alpha;
        }*/
    }
}
