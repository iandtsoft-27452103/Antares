using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Antares.AlphaBeta;
using static Antares.AttacksOperation;
using static Antares.Board;
using static Antares.Common;
using static Antares.Evaluate;
using static Antares.Feature2;
using static Antares.GenMoves;
using static Antares.Mate1Ply;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Antares
{
    internal class Playout
    {
        const int input_num = 1170;
        const int middle_num = 32;
        const int output_num = 1;
        const string model_file_name = "model_tp.pth";
        public static Device d = torch.device(DeviceType.CPU);
        public static TorchSharp.Modules.Linear input_layer = Linear(input_num, middle_num, device: d);
        public static TorchSharp.Modules.BatchNorm1d input_norm = BatchNorm1d(middle_num, 2e-05, device: d);
        public static TorchSharp.Modules.Linear output_layer = Linear(middle_num, output_num, device: d);
        public static TorchSharp.Modules.Sequential seq = Sequential(("input_layer", input_layer), ("input_norm", input_norm), ("input_relu", ReLU()), ("output_layer", output_layer), ("output_sigmoid", nn.Sigmoid()));

        public static void LoadTransProbs()
        {
            seq.load(model_file_name);
            seq.eval();
        }

        public static bool IsMoveValid(ref BoardTree bt, Move move, int color)
        {
            int ifrom = move.From;
            int ito = move.To;
            Piece cap_pc = move.CapPiece;
            // 玉の捕獲
            if (cap_pc == Piece.King)
                return false;
            Do(ref bt, move, color);
            // 手を指したら自玉がDiscovered Checkになった場合。
            if (IsAttacked(bt, bt.SQ_King[color], color) > 0)
            {
                UnDo(ref bt, move, color);
                return false;
            }
            //
            if (ifrom < Square_NB)
            {
                int idirec = (int)Adirec[ifrom, ito];
                // PIN駒を動かしてしまった場合。
                if (IsPinnedOnKing(bt, ifrom, (Direction)idirec, color) > 0)
                {
                    UnDo(ref bt, move, color);
                    return false;
                }
            }
            else
            {
                // 駒打ちなのに駒を取る手だった場合。
                if (cap_pc != Piece.Empty)
                {
                    UnDo(ref bt, move, color);
                    return false;
                }
            }
            UnDo(ref bt, move, color);
            return true;
        }

        // ロールアウトポリシーありのプレイアウト。
        // ロールアウトポリシーの計算が重すぎて、実用に耐えない…。
        public static int PlayOutUseRolloutPolicy(ref BoardTree bt, int start_color, int temp_ply, Move prev_move)
        {
            const int ply_max = 384;
            int result = 2;// 初期値は引き分け
            int ply = temp_ply;
            int color = start_color;

            while (ply < ply_max)
            {
                List<Move> moves = new List<Move>();

                if (IsAttacked(bt, bt.SQ_King[color], color) == 0)
                {
                    Move mate_move = MateIn1Ply(bt, color);

                    // 詰みありの場合
                    if (mate_move.Value != 0)
                    {
                        //ms.Add(mate_move);
                        if (color == (int)bt.RootColor)
                        {
                            result = 0;// root手番の勝ち
                        }
                        else
                        {
                            result = 1;// 相手の勝ち
                        }
                        break;
                    }

                    // 宣言勝ちできるかどうかの確認
                    if (IsAttacked(bt, bt.SQ_King[color ^ 1], color ^ 1) == 0)
                    {
                        // 先後両方の玉に王手がかかっていない場合のみ宣言勝ちできる
                        int iret = IsDeclarationWin(bt);
                        if (iret == 1 && color == (int)bt.RootColor && color == 0)
                        {
                            result = 0; // root手番の勝ち（かつroot手番は先手）
                            break;
                        }
                        else if (iret == 2 && color == (int)bt.RootColor && color == 1)
                        {
                            result = 1; // 相手の勝ち（かつroot手番は後手）
                            break;
                        }
                    }
                    GenDrop(bt, color, ref moves);
                    GenNoCap(bt, color, ref moves);
                    GenCap(bt, color, ref moves);
                }
                else
                {
                    GenEvasion(bt, color, ref moves);
                }

                List<Move> legal_moves = new List<Move>();

                for (int i = 0; i < moves.Count; i++)
                {
                    if (IsMoveValid(ref bt, moves[i], color))
                        legal_moves.Add(moves[i]);
                }

                // 合法手がなかったら終局
                if (legal_moves.Count == 0)
                {
                    if (color != (int)bt.RootColor)
                    {
                        result = 0;// root手番の勝ち
                    }
                    else
                    {
                        result = 1;// 相手の勝ち
                    }
                    break;
                }

                
                Random random = new Random();
                int a = random.Next(0, 9);
                if (a == 0)
                {
                    // 10%の確率でランダムに手を選んで指す。
                    int index = random.Next(0, legal_moves.Count - 1);
                    Do(ref bt, legal_moves[index], color);
                }
                else
                {
                    int move_count = legal_moves.Count;
                    torch.Tensor x = torch.zeros(move_count, input_num, dtype: torch.float32, device: d);
                    int[] colors = new int[move_count];
                    Move[] prev_moves = new Move[move_count];
                    for (int k = 0; k < move_count; k++)
                        colors[k] = color;

                    if (bt.ply == 1)
                    {
                        for (int k = 0; k < move_count; k++)
                            prev_moves[k] = new Move();
                    }
                    else
                    {
                        for (int k = 0; k < move_count; k++)
                            prev_moves[k] = prev_move;
                    }

                    List<double[]> li_ft = TransProbs6.MakeInputFeatures(bt, colors, legal_moves, prev_moves);
                    var ar_ft = li_ft.ToArray();
                    for (int k = 0; k < ar_ft.Length; k++)
                        x[k] = torch.tensor(ar_ft[k]);

                    double[] trans_probs = new double[move_count];
                    var y = seq.forward(x);
                    for (int k = 0; k < ar_ft.Length; k++)
                        trans_probs[k] = y[k].ToDouble();

                    int index;
                    int limit;
                    if (move_count <= 3)
                    {
                        index = 0;
                        limit = 1;
                    }
                    else
                    {
                        index = random.Next(0, 2);
                        limit = 3;
                    }
                    List<int> idxes = new List<int>();
                    for (int k = 0; k < limit; k++)
                    {
                        double v = trans_probs.Max();
                        int temp_index = Array.IndexOf(trans_probs, v);
                        idxes.Add(temp_index);
                        trans_probs[temp_index] = double.MinValue;
                    }
                    Do(ref bt, legal_moves[idxes[index]], color);
                }

                //ms.Add(legal_moves[index]);

                color ^= 1;

                ply++;
            }

            return result;
        }

        public static int PlayOutNoSearch(ref BoardTree bt, int start_color, int temp_ply, ref List<Move> ms)
        {
            const int ply_max = 384;
            int result = 2;// 初期値は引き分け
            int ply = temp_ply;
            int color = start_color;

            while (ply < ply_max)
            {
                List<Move> moves = new List<Move>();

                if (IsAttacked(bt, bt.SQ_King[color], color) == 0)
                {
                    Move mate_move = MateIn1Ply(bt, color);

                    // 詰みありの場合
                    if (mate_move.Value != 0)
                    {
                        ms.Add(mate_move);
                        if (color == (int)bt.RootColor)
                        {
                            result = 0;// root手番の勝ち
                        }
                        else
                        {
                            result = 1;// 相手の勝ち
                        }
                        break;
                    }

                    // 宣言勝ちできるかどうかの確認
                    if (IsAttacked(bt, bt.SQ_King[color ^ 1], color ^ 1) == 0)
                    {
                        // 先後両方の玉に王手がかかっていない場合のみ宣言勝ちできる
                        int iret = IsDeclarationWin(bt);
                        if (iret == 1 && color == (int)bt.RootColor && color == 0)
                        {
                            result = 0; // root手番の勝ち（かつroot手番は先手）
                            break;
                        }
                        else if (iret == 2 && color == (int)bt.RootColor && color == 1)
                        {
                            result = 1; // 相手の勝ち（かつroot手番は後手）
                            break;
                        }
                    }
                    GenDrop(bt, color, ref moves);
                    GenNoCap(bt, color, ref moves);
                    GenCap(bt, color, ref moves);
                }
                else
                {
                    GenEvasion(bt, color, ref moves);
                }

                List<Move> legal_moves = new List<Move>();

                for (int i = 0; i < moves.Count; i++)
                {
                    if (IsMoveValid(ref bt, moves[i], color))
                        legal_moves.Add(moves[i]);
                }

                // 合法手がなかったら終局
                if (legal_moves.Count == 0)
                {
                    if (color != (int)bt.RootColor)
                    {
                        result = 0;// root手番の勝ち
                    }
                    else
                    {
                        result = 1;// 相手の勝ち
                    }
                    break;
                }

                // ランダムに手を選んで指す。
                Random random = new Random();
                int index = random.Next(0, legal_moves.Count - 1);
                
                Do(ref bt, legal_moves[index], color);

                ms.Add(legal_moves[index]);

                color ^= 1;

                ply++;
            }

            return result;
        }

        public static int PlayOutUseSearch(ref AlphaBetaTree abt, int start_color, int temp_ply, ref List<Move> ms)
        {
            const int ply_max = 384;
            int result = 2;// 初期値は引き分け
            int ply = temp_ply;
            int color = start_color;
            const int depth = 2;
            int value = 0;
            int iret = 0;

            while (ply < ply_max)
            {
                List<Move> moves = new List<Move>();

                if (IsAttacked(abt.bt, abt.bt.SQ_King[color], color) == 0)
                {
                    Move mate_move = new Move();

                    // 詰みありの場合
                    if (MateIn1Ply(abt.bt, color).Value != 0)
                    {
                        if (color == (int)abt.bt.RootColor)
                        {
                            result = 0;// root手番の勝ち
                        }
                        else
                        {
                            result = 1;// 相手の勝ち
                        }
                        break;
                    }

                    // 宣言勝ちできるかどうかの確認
                    if (IsAttacked(abt.bt, abt.bt.SQ_King[color ^ 1], color ^ 1) == 0)
                    {
                        // 先後両方の玉に王手がかかっていない場合のみ宣言勝ちできる
                        iret = IsDeclarationWin(abt.bt);
                        if (iret == 1 && color == (int)abt.bt.RootColor && color == 0)
                        {
                            result = 0; // root手番の勝ち（かつroot手番は先手）
                            break;
                        }
                        else if (iret == 2 && color == (int)abt.bt.RootColor && color == 1)
                        {
                            result = 1; // 相手の勝ち（かつroot手番は後手）
                            break;
                        }
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
                            break;
                        case 2:
                            value = Value_Max;
                            break;
                    }

                    GenDrop(abt.bt, color, ref moves);
                    GenNoCap(abt.bt, color, ref moves);
                    GenCap(abt.bt, color, ref moves);
                }
                else
                {
                    GenEvasion(abt.bt, color, ref moves);
                }

                List<Move> legal_moves = new List<Move>();

                for (int i = 0; i < moves.Count; i++)
                {
                    if (IsMoveValid(ref abt.bt, moves[i], color))
                        legal_moves.Add(moves[i]);
                }

                // 合法手がなかったら終局
                if (legal_moves.Count == 0)
                {
                    if (color != (int)abt.bt.RootColor)
                    {
                        result = 0;// root手番の勝ち
                    }
                    else
                    {
                        result = 1;// 相手の勝ち
                    }
                    break;
                }

                // ※ソートを入れる。

                // 浅い探索を行う。

                uint state_node_new = (uint)AlphaBetaTree.NodeState.node_pv | (uint)AlphaBetaTree.NodeState.node_do_null_move
                    | (uint)AlphaBetaTree.NodeState.node_do_delta | (uint)AlphaBetaTree.NodeState.node_do_razoring
                    | (uint)AlphaBetaTree.NodeState.node_do_futility | (uint)AlphaBetaTree.NodeState.node_do_probcut;

                int static_eval = Eval(abt.bt);// 差分計算にすべきか？

                List<Move> temp_pv = new List<Move>();

                // ※全合法手は探索しない。
                /*for (int i = 0; i < legal_moves.Count; i++)
                {
                    int value = Search(ref abt, color, Value_Min, Value_Max, Ply_Inc * depth, 1, state_node_new, static_eval);
                }*/

                value = Search(ref abt, color, Value_Min, Value_Max, Ply_Inc * depth, 1, state_node_new, static_eval, ref temp_pv);

                if (value == Value_Draw && abt.pv.Count == 0)
                    break;

                Move best_move = abt.pv[0];

                ms.Add(best_move);

                abt.pv.Clear();

                Do(ref abt.bt, best_move, color);// indexのところは後で直す。

                color ^= 1;

                ply++;
            }

            return result;
        }

        public static double Sigmoid(double x)
        {
            return 1.0f / (1.0f + Math.Exp(-x));
        }

        public static float PlayOutShallow(ref BoardTree bt, int start_color, int temp_ply, ref List<Move> ms, int current_value)
        {
            const int ply_add = 8;// n手先の報酬を得たい。
            const float discounted_rate = 0.9f;
            int ply = temp_ply;
            int ply_max = temp_ply + ply_add;
            int color = start_color;
            List<int> li_value = new List<int>();
            float reward = 0.0f;
            bt.EvalArray[ply] = current_value;

            try
            {
                while (ply < ply_max)
                {
                    List<Move> moves = new List<Move>();

                    if (IsAttacked(bt, bt.SQ_King[color], color) == 0)
                    {
                        Move mate_move = MateIn1Ply(bt, color);

                        // 詰みありの場合
                        if (mate_move.Value != 0)
                        {
                            ms.Add(mate_move);
                            if (color == (int)bt.RootColor)
                            {
                                li_value.Add(Value_Max);// root手番の勝ち
                            }
                            else
                            {
                                li_value.Add(Value_Min);// 相手の勝ち
                            }
                            break;
                        }

                        // 宣言勝ちできるかどうかの確認
                        if (IsAttacked(bt, bt.SQ_King[color ^ 1], color ^ 1) == 0)
                        {
                            // 先後両方の玉に王手がかかっていない場合のみ宣言勝ちできる
                            int iret = IsDeclarationWin(bt);
                            if (iret == 1 && color == (int)bt.RootColor && color == 0)
                            {
                                li_value.Add(Value_Max); // root手番の勝ち（かつroot手番は先手）
                                break;
                            }
                            else if (iret == 2 && color == (int)bt.RootColor && color == 1)
                            {
                                li_value.Add(Value_Min); // 相手の勝ち（かつroot手番は後手）
                                break;
                            }
                        }
                        GenDrop(bt, color, ref moves);
                        GenNoCap(bt, color, ref moves);
                        GenCap(bt, color, ref moves);
                    }
                    else
                    {
                        GenEvasion(bt, color, ref moves);// ここでたまにエラーが起こる。原因は分かっていない。
                    }

                    List<Move> legal_moves = new List<Move>();

                    for (int i = 0; i < moves.Count; i++)
                    {
                        if (IsMoveValid(ref bt, moves[i], color))
                            legal_moves.Add(moves[i]);
                    }

                    // 合法手がなかったら終局
                    if (legal_moves.Count == 0)
                    {
                        if (color != (int)bt.RootColor)
                        {
                            li_value.Add(Value_Max);// root手番の勝ち
                        }
                        else
                        {
                            li_value.Add(Value_Min);// 相手の勝ち
                        }
                        break;
                    }

                    // ランダムに手を選んで指す。
                    Random random = new Random();
                    int index = random.Next(0, legal_moves.Count - 1);

                    Do(ref bt, legal_moves[index], color);

                    ms.Add(legal_moves[index]);

                    color ^= 1;

                    // ※ ここでリストの参照エラーが起こっている。
                    int v = (color == 0) ? EvalWrapper(bt, color ^ 1, ply, legal_moves[index], false) : -EvalWrapper(bt, color ^ 1, ply, legal_moves[index], false);
                    int v2 = Eval(bt);
                    if (color != start_color)// ※ここで反転ミスがあるかもしれない。
                        v = -v;
                    bt.EvalArray[++ply] = v;
                    li_value.Add(v);

                    //ply++;
                }

                //int v = (color == 0) ? Eval(bt) : -Eval(bt);

                float dr = discounted_rate;
                reward = (float)Sigmoid(current_value / 600);
                for (int i = 0; i < li_value.Count; i++)
                {
                    reward += dr * (float)Sigmoid(li_value[i] / 600);
                    dr *= dr;
                }

            }
            catch
            {
                reward = 0.0f;
            }

            return reward;
        }
    }
}
