using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Antares.Board;
using static Antares.Move;
using static Antares.GenMoves;
using static Antares.Common;
using static Antares.AttacksOperation;

namespace Antares
{
    // 詰み探索。かなりシンプルに書けたと思います。
    public class Mate
    {
        public Move[] move_cur;
        public List<List<Move>> mate_proc = new List<List<Move>>();
        public List<List<Move>> no_mate_proc = new List<List<Move>>();
        public Move first_move = new Move();
        public Move second_move = new Move();
        public int max_ply = 0;
        public bool is_abort = false;
        public bool is_mate_root = false;
        public BoardTree BTree;
        public List<Move> RootCheckMoves = new List<Move>();
        public string root_str_pv = "";
        public int debug_number = 0;

        public List<Move> GenRootCheckMoves(ref BoardTree bt)
        {
            List<Move> checkMoves = new List<Move>();
            GenCheck(bt, (int)bt.RootColor, ref checkMoves);
            return checkMoves;
        }

        public void MateSearchWrapper(int depth_limit)
        {
            int depth_max = depth_limit;// 後で変える
            int rest_depth = 1;

            while (rest_depth < depth_max)
            {
                max_ply = rest_depth;
                move_cur = new Move[max_ply + 1];
                for (int i = 0; i < max_ply + 1; i++)
                    move_cur[i] = new Move();

                mate_proc.Clear();
                no_mate_proc.Clear();
                first_move = new Move();
                second_move = new Move();

                is_mate_root = Offend(ref BTree, (int)BTree.RootColor, rest_depth, 1);

                if (is_mate_root)
                    break;

                if (is_abort)
                {
                    is_mate_root = false;
                    break;
                }

                rest_depth += 2;
            }

            if (is_mate_root && !is_abort)
            {
                Console.WriteLine("詰みあり");
                root_str_pv = OutResult(rest_depth);
            }
        }

        private string OutResult(int rest_depth)
        {
            int i, j, k;

            List<List<Move>> l = mate_proc;
            List<List<Move>> nl = no_mate_proc;

            bool b;
            int color;
            string str_pv = "";
            string[] str_color = new string[2];

            str_color[0] = "+";
            str_color[1] = "-";

            b = false;
            List<int> idxes = new List<int>();
            for (i = 0; i < l.Count; i++)
            {
                //if (is_abort)
                //return "";
                string s = (i + 1).ToString() + " / " + l.Count.ToString();
                Console.WriteLine(s);
                for (j = 0; j < nl.Count; j++)
                {
                    b = false;
                    for (k = 0; k < nl[j].Count; k++)
                    {
                        if (l[i][k].Value != nl[j][k].Value)
                        {
                            b = true;
                            break;
                        }
                    }
                    if (!b)
                    {
                        idxes.Add(i);
                    }
                }
            }

            for (i = 0; i < l.Count; i++)
            {
                //if (is_abort)
                //return "";
                if (idxes.Contains(i))
                    continue;
                str_pv = "";
                color = (int)BTree.RootColor;
                for (j = 0; j < rest_depth; j++)
                {
                    str_pv += str_color[color];
                    str_pv += CSA.Move2CSA(l[i][j]);
                    if (j != rest_depth - 1)
                        str_pv += ", ";
                    color ^= 1;
                }
                Console.WriteLine(str_pv);
            }

            return str_pv;
        }

        public bool Offend(ref BoardTree bt, int color, int rest_depth, int ply)// PrevMoveは要るか？
        {
            bool is_mate;
            List<Move> checkMoves = new List<Move>();

            if (is_abort)
                return false;

            // 王手を生成する
            if (ply != 1)
            {
                GenCheck(bt, color, ref checkMoves);
            }
            else
            {
                checkMoves = RootCheckMoves;
            }

            // 王手が生成されなかったら詰まない
            if (checkMoves.Count == 0)
            {
                return false;
            }

            for (int i = 0; i < checkMoves.Count; i++)
            {
                if (is_abort)
                    return false;

                move_cur[ply] = checkMoves[i];

                if (move_cur[ply].CapPiece == Piece.King)
                {
                    continue;
                }

                if (move_cur[ply].PieceType == Piece.Empty)// たまにこのような状態になる。原因は特定できていない。
                    continue;

                Do(ref bt, move_cur[ply], color);

                // // たまにこのような状態になる。原因は特定できていない。
                if (IsAttacked(bt, bt.SQ_King[color ^ 1], color ^ 1) == 0)
                {
                    UnDo(ref bt, move_cur[ply], color);
                    continue;
                }

                // Discoverd Checkになってしまった場合
                if (IsAttacked(bt, bt.SQ_King[color], color) != 0)
                {
                    UnDo(ref bt, move_cur[ply], color);
                    continue;
                }

                is_mate = Defend(ref bt, color ^ 1, rest_depth - 1, ply + 1);

                if (is_mate)
                {
                    if (ply == max_ply)
                    {
                        List<Move> moves = new List<Move>();
                        for (int j = 1; j < ply + 1; j++)
                            moves.Add(move_cur[j]);
                        mate_proc.Add(moves);
                    }
                    UnDo(ref bt, move_cur[ply], color);
                    return true;
                }

                /*if (ply == 13)
                {
                    int a = 0;
                }*/
                UnDo(ref bt, move_cur[ply], color);
            }

            return false;
        }

        public bool Defend(ref BoardTree bt, int color, int rest_depth, int ply)// PrevMoveは要るか？
        {
            bool is_mate;
            int mate_count = 0;
            List<Move> evasionMoves = new List<Move>();

            if (is_abort)
                return false;

            // 王手を避ける手を生成する
            GenEvasion(bt, color, ref evasionMoves);


            // 残り深さが1で王手を避ける手が生成されたら詰みではない
            if (rest_depth == 0 && evasionMoves.Count > 0)
            {
                return false;
            }

            // 王手を避ける手が生成されなかったら詰みである
            if (evasionMoves.Count == 0)
            {
                return true;
            }

            for (int i = 0; i < evasionMoves.Count; i++)
            {
                if (is_abort)
                    return false;

                move_cur[ply] = evasionMoves[i];

                Do(ref bt, move_cur[ply], color);


                if (IsAttacked(bt, bt.SQ_King[color], color) != 0)
                {
                    UnDo(ref bt, move_cur[ply], color);
                    continue;
                }

                is_mate = Offend(ref bt, color ^ 1, rest_depth - 1, ply + 1);


                if (!is_mate)
                {
                    List<Move> moves = new List<Move>();
                    for (int j = 0; j < ply - 1; j++)
                    {
                        moves.Add(move_cur[j + 1]);
                    }
                    no_mate_proc.Add(moves);

                    // ※守備側で不詰みがあった場合、1つ上の攻撃側の手までの手順は全部不詰みとなる。
                    UnDo(ref bt, move_cur[ply], color);
                    return false;
                }
                else
                {
                    mate_count++;
                }

                UnDo(ref bt, move_cur[ply], color);
            }

            if (ply == 2 && mate_count == evasionMoves.Count)
            {
                first_move = move_cur[1];
                second_move = move_cur[2];
            }

            return true;// どの手を指しても詰みだった場合
        }
    }
}
