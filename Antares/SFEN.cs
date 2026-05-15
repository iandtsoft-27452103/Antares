using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Antares.Common;
using static Antares.Hash;
using static Antares.Board;
using static System.Runtime.InteropServices.JavaScript.JSType;
using BitBoard = System.UInt128;
using Rand = System.UInt128;
using System.Net.NetworkInformation;

namespace Antares
{
    public class SFEN
    {
        public static string ToSFEN(BoardTree bt, int color)
        {
            int i, j, k, empty_count, num;
            string str_piece;
            string str_sfen = "";
            bool flag = false;
            i = 0;
            empty_count = 0;

            while (i < Square_NB)
            {
                str_piece = Str_SFEN_Pc[bt.Board[i]];
                if (str_piece == "")
                {
                    empty_count++;
                    flag = true;
                }
                else
                {
                    if (flag == true)
                    {
                        flag = false;
                        str_sfen += empty_count.ToString();
                        empty_count = 0;
                    }
                    str_sfen += str_piece;
                }
                if (i != (Square_NB - 1) && FileTable[i] == Common.File.File9)
                {
                    if (empty_count > 0)
                    {
                        flag = false;
                        str_sfen += empty_count.ToString();
                        empty_count = 0;
                    }
                    str_sfen += "/";
                }
                i++;
            }

            str_sfen += " ";
            str_sfen += Str_Color[color];
            str_sfen += " ";

            k = 0;
            if (bt.Hand[(int)Common.Color.Black] == 0 && bt.Hand[(int)Common.Color.White] == 0)
            {
                str_sfen += "-";
            }
            else
            {
                for (i = (int)Common.Color.Black; i < Color_NB; i++)
                {
                    for (j = (int)Piece.Rook; j >= (int)Piece.Pawn; j--)
                    {
                        num = (bt.Hand[i] & Hand_Mask[j]) >> Hand_Rev_Bit[j];
                        if (num == 0)
                            continue;
                        if (num > 0)
                        {
                            if (num == 1)
                            {
                                k = -Sign_Table[i] * j;
                                str_sfen += Str_SFEN_Pc[k];
                            }
                            else if (num > 1)
                            {
                                k = -Sign_Table[i] * j;
                                str_sfen += num.ToString() + Str_SFEN_Pc[k];
                            }
                        }
                    }
                }
            }

            str_sfen += " 1";
            return str_sfen;
        }

        public static BoardTree ToBoard(string str_sfen)
        {
            int /*i,*/ j, k, sq, limit, empty_num, int_pc, num;
            int color;
            bool flag = false;
            BoardTree bt = new BoardTree();
            BoardTreeAlloc(ref bt);
            Clear(ref bt);
            //i = 0;
            int_pc = 0;

            string[] str_temp = str_sfen.Split(' ');
            string str_board = str_temp[0];
            limit = (int)str_board.Length;
            sq = 0;
            for (j = 0; j < limit; j++)
            {
                string s = str_board.Substring(j, 1);
                if (s == "+")
                {
                    flag = true;
                }
                else if (s == "/")
                {
                    continue;
                }
                else
                {
                    if (Set_Empty_Num.Contains(s))
                    {
                        empty_num = Int_Empty_Num[s];
                        k = 0;
                        while (k < empty_num)
                        {
                            bt.Board[sq] = (int)Piece.Empty;
                            sq++;
                            k++;
                        }
                    }
                    else
                    {
                        int_pc = Int_Pc[s];
                        if (int_pc > 0)
                        {
                            if (flag == true)
                            {
                                int_pc += Promote;
                                flag = false;
                            }
                            bt.BB_Piece[(int)Common.Color.Black, int_pc] |= ABB_Mask[sq];
                            bt.BB_Occupied[(int)Common.Color.Black] |= ABB_Mask[sq];
                            if (int_pc == (int)Piece.King)
                            {
                                bt.SQ_King[(int)Common.Color.Black] = sq;
                            }
                        }
                        else
                        {
                            if (flag == true)
                            {
                                int_pc -= Promote;
                                flag = false;
                            }
                            bt.BB_Piece[(int)Common.Color.White, -int_pc] |= ABB_Mask[sq];
                            bt.BB_Occupied[(int)Common.Color.White] |= ABB_Mask[sq];
                            if (int_pc == -(int)Piece.King)
                            {
                                bt.SQ_King[(int)Common.Color.White] = sq;
                            }
                        }
                        bt.Board[sq] = int_pc;
                        sq += 1;
                    }
                }
            }
            string str_color = str_temp[1];
            bt.RootColor = (Common.Color)Num_Color[str_color];
            string str_hand = str_temp[2];
            limit = str_hand.Length;
            flag = false;
            num = 1;
            for (j = 0; j < limit; j++)
            {
                string s = str_hand.Substring(j, 1);
                if (s == "-")
                    break;
                if (s == "1" && !flag)
                {
                    flag = true;
                }
                else
                {
                    if (flag)
                    {
                        num = 10 + Int_Hand_Num[s];
                        flag = false;
                    }
                    else
                    {
                        if (Set_Hand_Num.Contains(s))
                        {
                            num = Int_Hand_Num[s];
                        }
                        else
                        {
                            int_pc = Int_Pc[s];
                            if (int_pc > 0)
                            {
                                color = (int)Common.Color.Black;
                            }
                            else
                            {
                                color = (int)Common.Color.White;
                                int_pc = -int_pc;
                            }
                            k = 0;
                            while (k < num)
                            {
                                bt.Hand[color] += Hand_Hash[int_pc];
                                k++;
                            }
                            num = 1;
                        }
                    }
                }
            }
            bt.CurrentHash = HashFunc(bt);
            bt.Hash[0] = bt.PrevHash;
            bt.Hash[1] = bt.CurrentHash;
            bt.ply = 1;
            return bt;
        }
    }
}
