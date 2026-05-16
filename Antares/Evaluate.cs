using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Antares.BitOperation;
using static Antares.Board;
using static Antares.Common;
using static Antares.Feature2;
using BitBoard = System.UInt128;

namespace Antares
{
    internal class Evaluate
    {
        public static int MakeListWithKKP(BoardTree bt, ref List<int> li0, ref List<int> li1)
        {
            int i, j, color;
            int sq_bk0 = bt.SQ_King[0];
            int sq_wk0 = bt.SQ_King[1];
            int sq_bk1 = Rev_Sq[sq_bk0];
            int sq_wk1 = Rev_Sq[sq_wk0];
            int kkp_score = 0;
            for (color = 0; color < Color_NB; color++)
            {
                for (i = (int)Piece.Pawn; i <= (int)Piece.Rook; i++)
                {
                    int n_hand = (bt.Hand[color] & Hand_Mask[i]) >> Hand_Rev_Bit[i];
                    for (j = 0; j < n_hand; j++)
                    {
                        int index = kkp_hand_start_index[i] + j;
                        kkp_score += (color == 0) ? fv_kkp[sq_bk0, sq_wk0, index] : -fv_kkp[sq_bk1, sq_wk1, index];
                        li0.Add(pp_hand_start_index[color, i] + j);
                        li1.Add(pp_hand_start_index[color ^ 1, i] + j);
                    }
                }
            }
            for (color = 0; color < Color_NB; color++)
            {
                for (i = (int)Piece.Pawn; i <= (int)Piece.Dragon; i++)
                {
                    if (i == (int)Piece.None || i == (int)Piece.King)
                        continue;
                    BitBoard bb = bt.BB_Piece[color, i];
                    while (bb != 0)
                    {
                        int sq = Square(bb);
                        bb ^= ABB_Mask[sq];
                        kkp_score += (color == 0) ? fv_kkp[sq_bk0, sq_wk0, kkp_index_table[color, i, sq]] : -fv_kkp[sq_bk1, sq_wk1, kkp_index_table[color, i, Rev_Sq[sq]]];
                        li0.Add(pp_index_table[color, i, sq]);
                        li1.Add(pp_index_table[color ^ 1, i, Rev_Sq[sq]]);
                    }
                }
            }
            return kkp_score;
        }

        public static int CalcKKP(BoardTree bt)
        {
            int i, j, color;
            int sq_bk0 = bt.SQ_King[0];
            int sq_wk0 = bt.SQ_King[1];
            int sq_bk1 = Rev_Sq[sq_bk0];
            int sq_wk1 = Rev_Sq[sq_wk0];
            int kkp_score = 0;
            for (color = 0; color < Color_NB; color++)
            {
                for (i = (int)Piece.Pawn; i <= (int)Piece.Rook; i++)
                {
                    int n_hand = (bt.Hand[color] & Hand_Mask[i]) >> Hand_Rev_Bit[i];
                    for (j = 0; j < n_hand; j++)
                    {
                        int index = kkp_hand_start_index[i] + j;
                        kkp_score += (color == 0) ? fv_kkp[sq_bk0, sq_wk0, index] : -fv_kkp[sq_bk1, sq_wk1, index];
                    }
                }
            }
            for (color = 0; color < Color_NB; color++)
            {
                for (i = (int)Piece.Pawn; i <= (int)Piece.Dragon; i++)
                {
                    if (i == (int)Piece.None || i == (int)Piece.King)
                        continue;
                    BitBoard bb = bt.BB_Piece[color, i];
                    while (bb != 0)
                    {
                        int sq = Square(bb);
                        bb ^= ABB_Mask[sq];
                        kkp_score += (color == 0) ? fv_kkp[sq_bk0, sq_wk0, kkp_index_table[color, i, sq]] : -fv_kkp[sq_bk1, sq_wk1, kkp_index_table[color, i, Rev_Sq[sq]]];
                    }
                }
            }
            return kkp_score;
        }

        public static void MakeList(BoardTree bt, ref List<int> li0, ref List<int> li1)
        {
            int i, j, color;
            for (color = 0; color < Color_NB; color++)
            {
                for (i = (int)Piece.Pawn; i <= (int)Piece.Rook; i++)
                {
                    int n_hand = (bt.Hand[color] & Hand_Mask[i]) >> Hand_Rev_Bit[i];
                    for (j = 0; j < n_hand; j++)
                    {
                        //int index = kkp_hand_start_index[i] + j;
                        li0.Add(pp_hand_start_index[color, i] + j);
                        li1.Add(pp_hand_start_index[color ^ 1, i] + j);
                    }
                }
            }
            for (color = 0; color < Color_NB; color++)
            {
                for (i = (int)Piece.Pawn; i <= (int)Piece.Dragon; i++)
                {
                    if (i == (int)Piece.None || i == (int)Piece.King)
                        continue;
                    BitBoard bb = bt.BB_Piece[color, i];
                    while (bb != 0)
                    {
                        int sq = Square(bb);
                        bb ^= ABB_Mask[sq];
                        li0.Add(pp_index_table[color, i, sq]);
                        li1.Add(pp_index_table[color ^ 1, i, Rev_Sq[sq]]);
                    }
                }
            }
        }

        public static void MakeList2(BoardTree bt, ref List<int> li0)
        {
            int i, j, color;
            for (color = 0; color < Color_NB; color++)
            {
                for (i = (int)Piece.Pawn; i <= (int)Piece.Rook; i++)
                {
                    int n_hand = (bt.Hand[color] & Hand_Mask[i]) >> Hand_Rev_Bit[i];
                    for (j = 0; j < n_hand; j++)
                    {
                        int index = kkp_hand_start_index[i] + j;
                        li0.Add(pp_hand_start_index[color, i] + j);
                    }
                }
            }
            for (color = 0; color < Color_NB; color++)
            {
                for (i = (int)Piece.Pawn; i <= (int)Piece.Dragon; i++)
                {
                    if (i == (int)Piece.None || i == (int)Piece.King)
                        continue;
                    BitBoard bb = bt.BB_Piece[color, i];
                    while (bb != 0)
                    {
                        int sq = Square(bb);
                        bb ^= ABB_Mask[sq];
                        li0.Add(pp_index_table[color, i, sq]);
                    }
                }
            }
        }

        // btには手を指した後の局面が入っているものと仮定する。colorはdrop_moveを指した方の手番。
        public static int CalcDiffDrop(BoardTree bt, int color, Move drop_move)
        {
            int s;
            const int piece_num = 38;
            int score = 0;
            int drop_sq = drop_move.To;
            int drop_piece = (int)drop_move.PieceType;
            int after_hand_num = (bt.Hand[color] & Hand_Mask[drop_piece]) >> Hand_Rev_Bit[drop_piece];
            int sq_bk = (color == 0) ? bt.SQ_King[0] : Rev_Sq[bt.SQ_King[0]];
            int sq_wk = (color == 0) ? bt.SQ_King[1] : Rev_Sq[bt.SQ_King[1]];
            int index, index2, pos;
            index = kkp_hand_start_index[drop_piece] + after_hand_num;// 手を指した後の枚数がn枚だったら、手を指す前の枚数はn + 1枚。
            score += (color == 0) ? -fv_kkp[sq_bk, sq_wk, index] : fv_kkp[sq_bk, sq_wk, index];
            index = (color == 0) ? kkp_index_table[color, drop_piece, drop_sq] : kkp_index_table[color, drop_piece, Rev_Sq[drop_sq]];
            score += (color == 0) ? fv_kkp[sq_bk, sq_wk, index] : -fv_kkp[sq_bk, sq_wk, index];
            List<int> li0 = new List<int>();
            List<int> li1 = new List<int>();
            MakeList(bt, ref li0, ref li1);// 1手指した後の駒リストが入っている。
            UnDo(ref bt, drop_move, color);
            List<int> li2 = new List<int>();
            List<int> li3 = new List<int>();
            MakeList(bt, ref li2, ref li3);// 1手指す前の駒リストが入っている。
            Do(ref bt, drop_move, color);

            sq_bk = bt.SQ_King[0];
            sq_wk = Rev_Sq[bt.SQ_King[1]];

            // 持ち駒の差分計算 => 1枚分減算する。
            index = pp_hand_start_index[color, drop_piece] + after_hand_num;
            pos = li2.IndexOf(index);
            for (s = 0; s <= pos; s++)
                score -= fv_kpp[sq_bk, index, li2[s]];
            for (s = pos + 1; s < li2.Count; s++)
                score -= fv_kpp[sq_bk, li2[s], index];
            index = pp_hand_start_index[color ^ 1, drop_piece] + after_hand_num;
            pos = li3.IndexOf(index);
            for (s = 0; s <= pos; s++)
                score += fv_kpp[sq_wk, index, li3[s]];
            for (s = pos + 1; s < li3.Count; s++)
                score += fv_kpp[sq_wk, li3[s], index];

            // 打った位置の差分計算
            index = pp_index_table[color, drop_piece, drop_sq];
            index2 = pp_index_table[color ^ 1, drop_piece, Rev_Sq[drop_sq]];
            pos = li0.IndexOf(index);
            for (s = 0; s <= pos; s++)
                score += fv_kpp[sq_bk, index, li0[s]];
            for (s = pos + 1; s < piece_num; s++)
                score += fv_kpp[sq_bk, li0[s], index];
            pos = li1.IndexOf(index2);
            for (s = 0; s <= pos; s++)
                score -= fv_kpp[sq_wk, index2, li1[s]];
            for (s = pos + 1; s < piece_num; s++)
                score -= fv_kpp[sq_wk, li1[s], index2];
            return score / FV_SCALE;
        }

        public static int CalcDiffNoCapNoPro(BoardTree bt, int color, Move move)
        {
            int s;
            const int piece_num = 38;
            int score = 0;
            int ifrom = move.From;
            int ito = move.To;
            int ipiece = (int)move.PieceType;
            int sq_bk = (color == 0) ? bt.SQ_King[0] : Rev_Sq[bt.SQ_King[0]];
            int sq_wk = (color == 0) ? bt.SQ_King[1] : Rev_Sq[bt.SQ_King[1]];
            int index, index2, pos;
            index = (color == 0) ? kkp_index_table[color, ipiece, ifrom] : kkp_index_table[color, ipiece, Rev_Sq[ifrom]];
            score -= (color == 0) ? fv_kkp[sq_bk, sq_wk, index] : -fv_kkp[sq_bk, sq_wk, index];
            index = (color == 0) ? kkp_index_table[color, ipiece, ito] : kkp_index_table[color, ipiece, Rev_Sq[ito]];
            score += (color == 0) ? fv_kkp[sq_bk, sq_wk, index] : -fv_kkp[sq_bk, sq_wk, index];
            List<int> li0 = new List<int>();
            List<int> li1 = new List<int>();
            MakeList(bt, ref li0, ref li1);// 1手指した後の駒リストが入っている。

            UnDo(ref bt, move, color);
            List<int> li2 = new List<int>();
            List<int> li3 = new List<int>();
            MakeList(bt, ref li2, ref li3);// 1手指す前の駒リストが入っている。

            Do(ref bt, move, color);

            sq_bk = bt.SQ_King[0];
            sq_wk = Rev_Sq[bt.SQ_King[1]];

            // 移動元の差分計算
            index = pp_index_table[color, ipiece, ifrom];
            index2 = pp_index_table[color ^ 1, ipiece, Rev_Sq[ifrom]];
            pos = li2.IndexOf(index);
            for (s = 0; s <= pos; s++)
                score -= fv_kpp[sq_bk, index, li2[s]];
            for (s = pos + 1; s < piece_num; s++)
                score -= fv_kpp[sq_bk, li2[s], index];
            pos = li3.IndexOf(index2);
            for (s = 0; s <= pos; s++)
                score += fv_kpp[sq_wk, index2, li3[s]];
            for (s = pos + 1; s < piece_num; s++)
                score += fv_kpp[sq_wk, li3[s], index2];

            // 移動先の差分計算
            index = pp_index_table[color, ipiece, ito];
            index2 = pp_index_table[color ^ 1, ipiece, Rev_Sq[ito]];
            pos = li0.IndexOf(index);
            for (s = 0; s <= pos; s++)
                score += fv_kpp[sq_bk, index, li0[s]];
            for (s = pos + 1; s < piece_num; s++)
                score += fv_kpp[sq_bk, li0[s], index];
            pos = li1.IndexOf(index2);
            for (s = 0; s <= pos; s++)
                score -= fv_kpp[sq_wk, index2, li1[s]];
            for (s = pos + 1; s < piece_num; s++)
                score -= fv_kpp[sq_wk, li1[s], index2];
            return score / FV_SCALE;
        }

        public static int CalcDiffNoCapPro(BoardTree bt, int color, Move move)
        {
            int s;
            const int piece_num = 38;
            int score = 0;
            int ifrom = move.From;
            int ito = move.To;
            int ipiece = (int)move.PieceType;
            int sq_bk = (color == 0) ? bt.SQ_King[0] : Rev_Sq[bt.SQ_King[0]];
            int sq_wk = (color == 0) ? bt.SQ_King[1] : Rev_Sq[bt.SQ_King[1]];
            int index, index2, pos;
            index = (color == 0) ? kkp_index_table[color, ipiece, ifrom] : kkp_index_table[color, ipiece, Rev_Sq[ifrom]];
            score -= (color == 0) ? fv_kkp[sq_bk, sq_wk, index] : -fv_kkp[sq_bk, sq_wk, index];
            index = (color == 0) ? kkp_index_table[color, ipiece + Promote, ito] : kkp_index_table[color, ipiece + Promote, Rev_Sq[ito]];
            score += (color == 0) ? fv_kkp[sq_bk, sq_wk, index] : -fv_kkp[sq_bk, sq_wk, index];
            List<int> li0 = new List<int>();
            List<int> li1 = new List<int>();
            MakeList(bt, ref li0, ref li1);// 1手指した後の駒リストが入っている。
            UnDo(ref bt, move, color);
            List<int> li2 = new List<int>();
            List<int> li3 = new List<int>();
            MakeList(bt, ref li2, ref li3);// 1手指す前の駒リストが入っている。
            Do(ref bt, move, color);

            sq_bk = bt.SQ_King[0];
            sq_wk = Rev_Sq[bt.SQ_King[1]];

            // 移動元の差分計算
            index = pp_index_table[color, ipiece, ifrom];
            index2 = pp_index_table[color ^ 1, ipiece, Rev_Sq[ifrom]];
            pos = li2.IndexOf(index);
            for (s = 0; s <= pos; s++)
                score -= fv_kpp[sq_bk, index, li2[s]];
            for (s = pos + 1; s < piece_num; s++)
                score -= fv_kpp[sq_bk, li2[s], index];
            pos = li3.IndexOf(index2);
            for (s = 0; s <= pos; s++)
                score += fv_kpp[sq_wk, index2, li3[s]];
            for (s = pos + 1; s < piece_num; s++)
                score += fv_kpp[sq_wk, li3[s], index2];

            ipiece += Promote;

            // 移動先の差分計算
            index = pp_index_table[color, ipiece, ito];
            index2 = pp_index_table[color ^ 1, ipiece, Rev_Sq[ito]];
            pos = li0.IndexOf(index);
            for (s = 0; s <= pos; s++)
                score += fv_kpp[sq_bk, index, li0[s]];
            for (s = pos + 1; s < piece_num; s++)
                score += fv_kpp[sq_bk, li0[s], index];
            pos = li1.IndexOf(index2);
            for (s = 0; s <= pos; s++)
                score -= fv_kpp[sq_wk, index2, li1[s]];
            for (s = pos + 1; s < piece_num; s++)
                score -= fv_kpp[sq_wk, li1[s], index2];
            return score / FV_SCALE;
        }

        public static int CalcDiffCapNoPro(BoardTree bt, int color, Move move)
        {
            int s;
            const int piece_num = 38;
            int score = 0;
            int ifrom = move.From;
            int ito = move.To;
            int ipiece = (int)move.PieceType;
            int icap_pc = (int)move.CapPiece;
            int kkp_icap_pc = (icap_pc < (int)Piece.King) ? icap_pc : icap_pc - Promote;
            int hand_num = (bt.Hand[color] & Hand_Mask[kkp_icap_pc]) >> Hand_Rev_Bit[kkp_icap_pc];
            int sq_bk = (color == 0) ? bt.SQ_King[0] : Rev_Sq[bt.SQ_King[0]];
            int sq_wk = (color == 0) ? bt.SQ_King[1] : Rev_Sq[bt.SQ_King[1]];
            int sq_bk2 = (color == 0) ? Rev_Sq[bt.SQ_King[0]] : bt.SQ_King[0];
            int sq_wk2 = (color == 0) ? Rev_Sq[bt.SQ_King[1]] : bt.SQ_King[1];
            int index, index2, pos;
            index = (color == 0) ? kkp_index_table[color, ipiece, ifrom] : kkp_index_table[color, ipiece, Rev_Sq[ifrom]];
            score -= (color == 0) ? fv_kkp[sq_bk, sq_wk, index] : -fv_kkp[sq_bk, sq_wk, index];
            index = (color == 0) ? kkp_index_table[color, ipiece, ito] : kkp_index_table[color, ipiece, Rev_Sq[ito]];
            score += (color == 0) ? fv_kkp[sq_bk, sq_wk, index] : -fv_kkp[sq_bk, sq_wk, index];
            index = (color == 0) ? kkp_index_table[color ^ 1, icap_pc, Rev_Sq[ito]] : kkp_index_table[color ^ 1, icap_pc, ito];
            score += (color == 0) ? fv_kkp[sq_bk2, sq_wk2, index] : -fv_kkp[sq_bk2, sq_wk2, index];
            index = kkp_hand_start_index[kkp_icap_pc] + hand_num - 1;
            score += (color == 0) ? fv_kkp[sq_bk, sq_wk, index] : -fv_kkp[sq_bk, sq_wk, index];
            List<int> li0 = new List<int>();
            List<int> li1 = new List<int>();
            MakeList(bt, ref li0, ref li1);// 1手指した後の駒リストが入っている。
            UnDo(ref bt, move, color);
            List<int> li2 = new List<int>();
            List<int> li3 = new List<int>();
            MakeList(bt, ref li2, ref li3);// 1手指す前の駒リストが入っている。

            sq_bk = bt.SQ_King[0];
            sq_wk = Rev_Sq[bt.SQ_King[1]];

            Do(ref bt, move, color);

            //Console.WriteLine("Edward Elgar");

            // 移動元の差分計算
            index = pp_index_table[color, ipiece, ifrom];
            index2 = pp_index_table[color ^ 1, ipiece, Rev_Sq[ifrom]];
            pos = li2.IndexOf(index);
            for (s = 0; s <= pos; s++)
            {
                score -= fv_kpp[sq_bk, index, li2[s]]; 
            }               
            for (s = pos + 1; s < piece_num; s++)
            {
                score -= fv_kpp[sq_bk, li2[s], index];
            }
                
            pos = li3.IndexOf(index2);
            for (s = 0; s <= pos; s++)
            {
                score += fv_kpp[sq_wk, index2, li3[s]];
            }
                
            for (s = pos + 1; s < piece_num; s++)
            {
                score += fv_kpp[sq_wk, li3[s], index2];
            }           

            int index_from0 = index;
            int index_from1 = index2;

            // 移動先の差分計算
            index = pp_index_table[color, ipiece, ito];
            index2 = pp_index_table[color ^ 1, ipiece, Rev_Sq[ito]];
            pos = li0.IndexOf(index);
            for (s = 0; s <= pos; s++)
            {
                score += fv_kpp[sq_bk, index, li0[s]];
            }
            for (s = pos + 1; s < piece_num; s++)
            {
                score += fv_kpp[sq_bk, li0[s], index];
            }
            pos = li1.IndexOf(index2);
            for (s = 0; s <= pos; s++)
            {
                score -= fv_kpp[sq_wk, index2, li1[s]];

            }
            for (s = pos + 1; s < piece_num; s++)
            {
                score -= fv_kpp[sq_wk, li1[s], index2];
            }

            int index_to0 = index;
            int index_to1 = index2;

            // 取られた駒の差分計算
            index = pp_index_table[color ^ 1, icap_pc, ito];
            index2 = pp_index_table[color, icap_pc, Rev_Sq[ito]];
            pos = li2.IndexOf(index);
            for (s = 0; s <= pos; s++)
            {
                if (li2[s] != index_from0)
                {
                    score -= fv_kpp[sq_bk, index, li2[s]];
                }
            }
            for (s = pos + 1; s < piece_num; s++)
            {
                if (li2[s] != index_from0)
                {
                    score -= fv_kpp[sq_bk, li2[s], index];
                }
            }
            pos = li3.IndexOf(index2);
            for (s = 0; s <= pos; s++)
            {
                if (li3[s] != index_from1)
                {
                    score += fv_kpp[sq_wk, index2, li3[s]];
                }
            }
            for (s = pos + 1; s < piece_num; s++)
            {
                if (li3[s] != index_from1)
                {
                    score += fv_kpp[sq_wk, li3[s], index2];
                }
            }

            // 駒台の差分計算 => 1枚分加算する。
            index = pp_hand_start_index[color, kkp_icap_pc] + hand_num - 1;
            index2 = pp_hand_start_index[color ^ 1, kkp_icap_pc] + hand_num - 1;
            pos = li0.IndexOf(index);
            for (s = 0; s <= pos; s++)
            {
                if (li0[s] != index_to0)
                {
                    score += fv_kpp[sq_bk, index, li0[s]];
                }
            }
            for (s = pos + 1; s < piece_num; s++)
            {
                if (li0[s] != index_to0)
                {
                    score += fv_kpp[sq_bk, li0[s], index];
                }
            }
            pos = li1.IndexOf(index2);
            for (s = 0; s <= pos; s++)
            {
                if (li1[s] != index_to1)
                {
                    score -= fv_kpp[sq_wk, index2, li1[s]];
                }
            }
            for (s = pos + 1; s < piece_num; s++)
            {
                if (li1[s] != index_to1)
                {
                    score -= fv_kpp[sq_wk, li1[s], index2];
                }
            }
            return score / FV_SCALE;
        }

        public static int CalcDiffCapPro(BoardTree bt, int color, Move move)
        {
            int s;
            const int piece_num = 38;
            int score = 0;
            int ifrom = move.From;
            int ito = move.To;
            int ipiece = (int)move.PieceType;
            int ipc_promo = ipiece + Promote;
            int icap_pc = (int)move.CapPiece;
            int kkp_icap_pc = (icap_pc < (int)Piece.King) ? icap_pc : icap_pc - Promote;
            int hand_num = (bt.Hand[color] & Hand_Mask[kkp_icap_pc]) >> Hand_Rev_Bit[kkp_icap_pc];
            int sq_bk = (color == 0) ? bt.SQ_King[0] : Rev_Sq[bt.SQ_King[0]];
            int sq_wk = (color == 0) ? bt.SQ_King[1] : Rev_Sq[bt.SQ_King[1]];
            int sq_bk2 = (color == 0) ? Rev_Sq[bt.SQ_King[0]] : bt.SQ_King[0];
            int sq_wk2 = (color == 0) ? Rev_Sq[bt.SQ_King[1]] : bt.SQ_King[1];
            int index, index2, pos;
            index = (color == 0) ? kkp_index_table[color, ipiece, ifrom] : kkp_index_table[color, ipiece, Rev_Sq[ifrom]];
            score -= (color == 0) ? fv_kkp[sq_bk, sq_wk, index] : -fv_kkp[sq_bk, sq_wk, index];
            index = (color == 0) ? kkp_index_table[color, ipc_promo, ito] : kkp_index_table[color, ipc_promo, Rev_Sq[ito]];
            score += (color == 0) ? fv_kkp[sq_bk, sq_wk, index] : -fv_kkp[sq_bk, sq_wk, index];
            index = (color == 0) ? kkp_index_table[color ^ 1, icap_pc, Rev_Sq[ito]] : kkp_index_table[color ^ 1, icap_pc, ito];
            score += (color == 0) ? fv_kkp[sq_bk2, sq_wk2, index] : -fv_kkp[sq_bk2, sq_wk2, index];
            index = kkp_hand_start_index[kkp_icap_pc] + hand_num - 1;
            score += (color == 0) ? fv_kkp[sq_bk, sq_wk, index] : -fv_kkp[sq_bk, sq_wk, index];
            List<int> li0 = new List<int>();
            List<int> li1 = new List<int>();
            MakeList(bt, ref li0, ref li1);// 1手指した後の駒リストが入っている。
            UnDo(ref bt, move, color);
            List<int> li2 = new List<int>();
            List<int> li3 = new List<int>();
            MakeList(bt, ref li2, ref li3);// 1手指す前の駒リストが入っている。

            sq_bk = bt.SQ_King[0];
            sq_wk = Rev_Sq[bt.SQ_King[1]];

            Do(ref bt, move, color);

            // 移動元の差分計算
            index = pp_index_table[color, ipiece, ifrom];
            index2 = pp_index_table[color ^ 1, ipiece, Rev_Sq[ifrom]];
            pos = li2.IndexOf(index);
            for (s = 0; s <= pos; s++)
            {
                score -= fv_kpp[sq_bk, index, li2[s]];
            }
            for (s = pos + 1; s < piece_num; s++)
            {
                score -= fv_kpp[sq_bk, li2[s], index];
            }

            pos = li3.IndexOf(index2);
            for (s = 0; s <= pos; s++)
            {
                score += fv_kpp[sq_wk, index2, li3[s]];
            }

            for (s = pos + 1; s < piece_num; s++)
            {
                score += fv_kpp[sq_wk, li3[s], index2];
            }

            int index_from0 = index;
            int index_from1 = index2;

            // 移動先の差分計算
            index = pp_index_table[color, ipc_promo, ito];
            index2 = pp_index_table[color ^ 1, ipc_promo, Rev_Sq[ito]];
            pos = li0.IndexOf(index);
            for (s = 0; s <= pos; s++)
            {
                score += fv_kpp[sq_bk, index, li0[s]];
            }
            for (s = pos + 1; s < piece_num; s++)
            {
                score += fv_kpp[sq_bk, li0[s], index];
            }
            pos = li1.IndexOf(index2);
            for (s = 0; s <= pos; s++)
            {
                score -= fv_kpp[sq_wk, index2, li1[s]];
            }
            for (s = pos + 1; s < piece_num; s++)
            {
                score -= fv_kpp[sq_wk, li1[s], index2];
            }

            int index_to0 = index;
            int index_to1 = index2;

            // 取られた駒の差分計算
            index = pp_index_table[color ^ 1, icap_pc, ito];
            index2 = pp_index_table[color, icap_pc, Rev_Sq[ito]];
            pos = li2.IndexOf(index);
            for (s = 0; s <= pos; s++)
            {
                if (li2[s] != index_from0)
                {
                    score -= fv_kpp[sq_bk, index, li2[s]];
                }
            }
            for (s = pos + 1; s < piece_num; s++)
            {
                if (li2[s] != index_from0)
                {
                    score -= fv_kpp[sq_bk, li2[s], index];
                }
            }
            pos = li3.IndexOf(index2);
            for (s = 0; s <= pos; s++)
            {
                if (li3[s] != index_from1)
                {
                    score += fv_kpp[sq_wk, index2, li3[s]];
                }
            }
            for (s = pos + 1; s < piece_num; s++)
            {
                if (li3[s] != index_from1)
                {
                    score += fv_kpp[sq_wk, li3[s], index2];
                }
            }

            // 駒台の差分計算 => 1枚分加算する。
            index = pp_hand_start_index[color, kkp_icap_pc] + hand_num - 1;
            index2 = pp_hand_start_index[color ^ 1, kkp_icap_pc] + hand_num - 1;
            pos = li0.IndexOf(index);
            for (s = 0; s <= pos; s++)
            {
                if (li0[s] != index_to0)
                {
                    score += fv_kpp[sq_bk, index, li0[s]];
                }
            }
            for (s = pos + 1; s < piece_num; s++)
            {
                if (li0[s] != index_to0)
                {
                    score += fv_kpp[sq_bk, li0[s], index];
                }
            }
            pos = li1.IndexOf(index2);
            for (s = 0; s <= pos; s++)
            {
                if (li1[s] != index_to1)
                {
                    score -= fv_kpp[sq_wk, index2, li1[s]];
                }
            }
            for (s = pos + 1; s < piece_num; s++)
            {
                if (li1[s] != index_to1)
                {
                    score -= fv_kpp[sq_wk, li1[s], index2];
                }
            }
            return score / FV_SCALE;
        }

        public static int Eval(BoardTree bt)
        {
            int i, j, k0, k1, l0, l1;
            const int pieces_sum = 38;
            const int fv_scale = 32;
            int sq_bk = bt.SQ_King[0];
            int sq_wk = Rev_Sq[bt.SQ_King[1]];
            List<int> li0 = new List<int>();
            List<int> li1 = new List<int>();
            int score = MakeListWithKKP(bt, ref li0, ref li1);
            for (i = 0; i < pieces_sum; i++)
            {
                k0 = li0[i];
                k1 = li1[i];
                for (j = 0; j <= i; j++)
                {
                    l0 = li0[j];
                    l1 = li1[j];
                    score += (int)fv_kpp[sq_bk, k0, l0];
                    score -= (int)fv_kpp[sq_wk, k1, l1];
                }
            }

            return score / fv_scale;
        }

        // ※反転漏れは多分あるので、要確認
        public static int EvalWrapper(BoardTree bt, int color, int ply, Move move, bool is_root)
        {
            int score = 0;
            int ifrom = move.From;
            if (is_root)
            {
                score = (color == 0) ? Eval(bt) : -Eval(bt);
            }
            else
            {
                //int ply = bt.ply - 1;
                //score = bt.EvalArray[ply];
                if (ifrom >= Square_NB)
                {
                    score = bt.EvalArray[ply];
                    score += (color == 0) ? CalcDiffDrop(bt, color, move) : -CalcDiffDrop(bt, color, move);
                }
                else
                {
                    int ipiece = (int)move.PieceType;
                    int icap_pc = (int)move.CapPiece;
                    int is_promo = move.FlagPromo;
                    if (ipiece == (int)Piece.King)
                    {
                        score = (color == 0) ? Eval(bt) : -Eval(bt);
                    }
                    else
                    {
                        if (icap_pc > 0)
                        {
                            if (is_promo == 1)
                            {
                                score = bt.EvalArray[ply];
                                score += (color == 0) ? CalcDiffCapPro(bt, color, move) : -CalcDiffCapPro(bt, color, move);
                            }
                            else
                            {
                                score = bt.EvalArray[ply];
                                score += (color == 0) ? CalcDiffCapNoPro(bt, color, move) : -CalcDiffCapNoPro(bt, color, move);
                            }
                        }
                        else
                        {
                            if (is_promo == 1)
                            {
                                score = bt.EvalArray[ply];
                                score += (color == 0) ? CalcDiffNoCapPro(bt, color, move) : -CalcDiffNoCapPro(bt, color, move);
                            }
                            else
                            {
                                score = bt.EvalArray[ply];
                                score += (color == 0) ? CalcDiffNoCapNoPro(bt, color, move) : -CalcDiffNoCapNoPro(bt, color, move);
                            }
                        }
                    }
                }
            }
            //bt.EvalArray[bt.ply] = (color == 0) ? score : -score;
            //bt.EvalArray[ply] = score;
            return score;
        }
    }
}
