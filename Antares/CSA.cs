using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.NetworkInformation;
using System.Text;
using System.Threading.Tasks;
using static Antares.Board;
using static Antares.Common;

namespace Antares
{
    public class CSA
    {
        public static Move CSA2Move(BoardTree bt, string str_csa)
        {
            Move move = new Move();
            int ifrom = Array.IndexOf(Str_CSA, str_csa.Substring(0, 2));
            int ito = Array.IndexOf(Str_CSA, str_csa.Substring(2, 2));
            Piece piece;
            int flag_promo = 0;
            if (ifrom < Square_NB)
            {
                piece = (Piece)Math.Abs(bt.Board[ifrom]);
            }
            else
            {
                piece = (Piece)CSA_TO_PC[str_csa.Substring(4, 2)];
                ifrom += (int)piece - 1;
            }

            Piece cap_piece = (Piece)Math.Abs(bt.Board[ito]);
            if (piece < Piece.King && CSA_TO_PC[str_csa.Substring(4, 2)] > (int)Piece.King)
            {
                flag_promo = 1;
            }
            move.Pack(ifrom, ito, piece, cap_piece, flag_promo);
            return move;
        }

        public static string Move2CSA(Move move)
        {
            string str;

            str = Str_CSA[move.From];
            str += Str_CSA[move.To];
            if (move.FlagPromo == 0)
            {
                str += Str_Piece[(int)move.PieceType];
            }
            else
            {
                str += Str_Piece[(int)move.PieceType + 8];
            }

            return str;
        }
    }
}
