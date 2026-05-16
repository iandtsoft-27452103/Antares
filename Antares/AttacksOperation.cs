using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BitBoard = System.UInt128;
using static Antares.Common;
using static Antares.BitOperation;
using static Antares.Board;
using System.Drawing;
using System.Net.NetworkInformation;

namespace Antares
{
    public class AttacksOperation
    {
        public static BitBoard IsPinnedOnKing(BoardTree bt, int sq, Direction idirec, int color)
        {
            BitBoard bb_occupied, bb_attacks;
            bb_occupied = bt.BB_Occupied[(int)Common.Color.Black] | bt.BB_Occupied[(int)Common.Color.White];
            switch (Math.Abs((int)idirec))
            {
                case (int)Direction.Direc_File_U2d:
                    bb_attacks = ABB_File_Attacks[sq][bb_occupied & ABB_File_Mask_Ex[sq]];
                    if ((bb_attacks & ABB_Mask[bt.SQ_King[color]]) > 0)
                        return bb_attacks & (bt.BB_Piece[color ^ 1, (int)Piece.Rook] | bt.BB_Piece[color ^ 1, (int)Piece.Dragon] | bt.BB_Piece[color ^ 1, (int)Piece.Lance]);
                    break;
                case (int)Direction.Direc_Rank_L2r:
                    bb_attacks = ABB_Rank_Attacks[sq][bb_occupied & ABB_Rank_Mask_Ex[sq]];
                    if ((bb_attacks & ABB_Mask[bt.SQ_King[color]]) > 0)
                        return bb_attacks & (bt.BB_Piece[color ^ 1, (int)Piece.Rook] | bt.BB_Piece[color ^ 1, (int)Piece.Dragon]);
                    break;
                case (int)Direction.Direc_Diag1_U2d:
                    bb_attacks = ABB_Diag1_Attacks[sq][bb_occupied & ABB_Diag1_Mask_Ex[sq]];
                    if ((bb_attacks & ABB_Mask[bt.SQ_King[color]]) > 0)
                        return bb_attacks & (bt.BB_Piece[color ^ 1, (int)Piece.Bishop] | bt.BB_Piece[color ^ 1, (int)Piece.Horse]);
                    break;
                case (int)Direction.Direc_Diag2_U2d:
                    bb_attacks = ABB_Diag2_Attacks[sq][bb_occupied & ABB_Diag2_Mask_Ex[sq]];
                    if ((bb_attacks & ABB_Mask[bt.SQ_King[color]]) > 0)
                        return bb_attacks & (bt.BB_Piece[color ^ 1, (int)Piece.Bishop] | bt.BB_Piece[color ^ 1, (int)Piece.Horse]);
                    break;
            }
            return 0;
        }

        public static BitBoard AttacksToPiece(BoardTree bt, int sq, int color)
        {
            BitBoard bb_ret, bb_occupied, bb_total_gold, bb_hdk, bb_bh, bb_rd, bb_lance_attacks;
            bb_occupied = bt.BB_Occupied[(int)Common.Color.Black] | bt.BB_Occupied[(int)Common.Color.White];
            bb_ret = bt.BB_Piece[color, (int)Piece.Pawn] & ABB_Piece_Attacks[color ^ 1, (int)Piece.Pawn, sq];
            bb_ret |= bt.BB_Piece[color, (int)Piece.Knight] & ABB_Piece_Attacks[color ^ 1, (int)Piece.Knight, sq];
            bb_ret |= bt.BB_Piece[color, (int)Piece.Silver] & ABB_Piece_Attacks[color ^ 1, (int)Piece.Silver, sq];
            bb_total_gold = bt.BB_Piece[color, (int)Piece.Gold] | bt.BB_Piece[color, (int)Piece.Pro_Pawn] | bt.BB_Piece[color, (int)Piece.Pro_Lance] | bt.BB_Piece[color, (int)Piece.Pro_Knight] | bt.BB_Piece[color, (int)Piece.Pro_Silver];
            bb_ret |= bb_total_gold & ABB_Piece_Attacks[color ^ 1, (int)Piece.Gold, sq];
            bb_hdk = bt.BB_Piece[color, (int)Piece.Horse] | bt.BB_Piece[color, (int)Piece.Dragon] | bt.BB_Piece[color, (int)Piece.King];
            bb_ret |= bb_hdk & ABB_Piece_Attacks[color ^ 1, (int)Piece.King, sq];
            bb_bh = bt.BB_Piece[color, (int)Piece.Bishop] | bt.BB_Piece[color, (int)Piece.Horse];
            bb_ret |= bb_bh & ABB_Diagonal_Attacks[sq][ABB_Diagonal_Mask_Ex[sq] & bb_occupied];
            bb_rd = bt.BB_Piece[color, (int)Piece.Rook] | bt.BB_Piece[color, (int)Piece.Dragon];
            bb_ret |= bb_rd & ABB_Cross_Attacks[sq][ABB_Cross_Mask_Ex[sq] & bb_occupied];
            bb_lance_attacks = ABB_Lance_Attacks[color ^ 1, sq][ABB_Lance_Mask_Ex[color ^ 1, sq] & bb_occupied];
            bb_ret |= bt.BB_Piece[color, (int)Piece.Lance] & bb_lance_attacks;
            return bb_ret;
        }

        public static bool IsMatePawnDrop(BoardTree bt, int sq_drop, int color)
        {
            BitBoard bb_sum, bb_occupied, bb_total_gold, bb_bh, bb_rd, bb_hd, bb_move;
            int ifrom, ito, iking;
            bool bret;
            if (color == (int)Common.Color.White)
            {
                if ((sq_drop - 9) >= 0 && bt.Board[sq_drop - 9] != -(int)Piece.King)
                {
                    return false;
                }
            }
            else
            {
                if ((sq_drop + 9) < Square_NB && bt.Board[sq_drop + 9] != (int)Piece.King)
                {
                    return false;
                }
            }
            bb_sum = bt.BB_Piece[color, (int)Piece.Knight] & ABB_Piece_Attacks[color ^ 1, (int)Piece.Knight, sq_drop];
            bb_sum |= bt.BB_Piece[color, (int)Piece.Silver] & ABB_Piece_Attacks[color ^ 1, (int)Piece.Silver, sq_drop];
            bb_total_gold = bt.BB_Piece[color, (int)Piece.Gold] | bt.BB_Piece[color, (int)Piece.Pro_Pawn] | bt.BB_Piece[color, (int)Piece.Pro_Lance] | bt.BB_Piece[color, (int)Piece.Pro_Knight] | bt.BB_Piece[color, (int)Piece.Pro_Silver];
            bb_sum |= bb_total_gold & ABB_Piece_Attacks[color ^ 1, (int)Piece.Gold, sq_drop];
            bb_occupied = bt.BB_Occupied[(int)Common.Color.Black] | bt.BB_Occupied[(int)Common.Color.White];
            bb_bh = bt.BB_Piece[color, (int)Piece.Bishop] | bt.BB_Piece[color, (int)Piece.Horse];
            bb_sum |= bb_bh & ABB_Diagonal_Attacks[sq_drop][ABB_Diagonal_Mask_Ex[sq_drop] & bb_occupied];
            bb_rd = bt.BB_Piece[color, (int)Piece.Rook] | bt.BB_Piece[color, (int)Piece.Dragon];
            bb_sum |= bb_rd & ABB_Cross_Attacks[sq_drop][ABB_Cross_Mask_Ex[sq_drop] & bb_occupied];
            bb_hd = bt.BB_Piece[color, (int)Piece.Horse] | bt.BB_Piece[color, (int)Piece.Dragon];
            bb_sum |= bb_hd & ABB_Piece_Attacks[color, (int)Piece.King, sq_drop];
            while (bb_sum != 0)
            {
                ifrom = Square(bb_sum);
                bb_sum ^= ABB_Mask[ifrom];
                if (IsDiscoverKing(bt, ifrom, sq_drop, color))
                {
                    continue;
                }
                return false;
            }
            iking = bt.SQ_King[color];
            bret = true;
            bt.BB_Occupied[color ^ 1] ^= ABB_Mask[sq_drop];
            bb_move = ABB_Piece_Attacks[color, (int)Piece.King, iking] & ~bt.BB_Occupied[color] & BB_Full;
            while (bb_move > 0)
            {
                ito = Square(bb_move);
                if (IsAttacked(bt, ito, color) == 0)
                {
                    bret = false;
                    break;
                }
                bb_move ^= ABB_Mask[ito];
            }
            bt.BB_Occupied[color ^ 1] ^= ABB_Mask[sq_drop];
            return bret;
        }

        public static BitBoard IsAttacked(BoardTree bt, int sq, int color)
        {
            BitBoard bb_ret, bb_occupied, bb_total_gold, bb_hdk, bb_bh, bb_rd, bb_lance_attacks;
            bb_ret = 0;
            bb_occupied = bt.BB_Occupied[(int)Common.Color.Black] | bt.BB_Occupied[(int)Common.Color.White];
            if ((sq + Delta_Table[color]) >= 0 && (sq + Delta_Table[color]) < Square_NB)
            {
                if (bt.Board[sq + Delta_Table[color]] == (Sign_Table[color] * (int)Piece.Pawn))
                {
                    bb_ret = ABB_Mask[sq + Delta_Table[color]];
                }
            }
            bb_ret |= bt.BB_Piece[color ^ 1, (int)Piece.Knight] & ABB_Piece_Attacks[color, (int)Piece.Knight, sq];
            bb_ret |= bt.BB_Piece[color ^ 1, (int)Piece.Silver] & ABB_Piece_Attacks[color, (int)Piece.Silver, sq];
            bb_total_gold = bt.BB_Piece[color ^ 1, (int)Piece.Gold] | bt.BB_Piece[color ^ 1, (int)Piece.Pro_Pawn] | bt.BB_Piece[color ^ 1, (int)Piece.Pro_Lance] | bt.BB_Piece[color ^ 1, (int)Piece.Pro_Knight] | bt.BB_Piece[color ^ 1, (int)Piece.Pro_Silver];
            bb_ret |= bb_total_gold & ABB_Piece_Attacks[color, (int)Piece.Gold, sq];
            bb_hdk = bt.BB_Piece[color ^ 1, (int)Piece.Horse] | bt.BB_Piece[color ^ 1, (int)Piece.Dragon] | bt.BB_Piece[color ^ 1, (int)Piece.King];
            bb_ret |= bb_hdk & ABB_Piece_Attacks[color, (int)Piece.King, sq];
            bb_bh = bt.BB_Piece[color ^ 1, (int)Piece.Bishop] | bt.BB_Piece[color ^ 1, (int)Piece.Horse];
            bb_ret |= bb_bh & ABB_Diagonal_Attacks[sq][ABB_Diagonal_Mask_Ex[sq] & bb_occupied];
            bb_rd = bt.BB_Piece[color ^ 1, (int)Piece.Rook] | bt.BB_Piece[color ^ 1, (int)Piece.Dragon];
            bb_ret |= bb_rd & ABB_Cross_Attacks[sq][ABB_Cross_Mask_Ex[sq] & bb_occupied];
            bb_lance_attacks = ABB_Lance_Attacks[color, sq][ABB_Lance_Mask_Ex[color, sq] & bb_occupied];
            bb_ret |= bt.BB_Piece[color ^ 1, (int)Piece.Lance] & bb_lance_attacks;
            return bb_ret;
        }

        public static BitBoard IsAttackedByLongPieces(BoardTree bt, int sq, int color)
        {
            BitBoard bb_ret, bb_occupied, bb_bh, bb_rd, bb_lance_attacks;
            bb_ret = 0;
            bb_occupied = bt.BB_Occupied[(int)Common.Color.Black] | bt.BB_Occupied[(int)Common.Color.White];
            bb_bh = bt.BB_Piece[color ^ 1, (int)Piece.Bishop] | bt.BB_Piece[color ^ 1, (int)Piece.Horse];
            bb_ret |= bb_bh & ABB_Diagonal_Attacks[sq][ABB_Diagonal_Mask_Ex[sq] & bb_occupied];
            bb_rd = bt.BB_Piece[color ^ 1, (int)Piece.Rook] | bt.BB_Piece[color ^ 1, (int)Piece.Dragon];
            bb_ret |= bb_rd & ABB_Cross_Attacks[sq][ABB_Cross_Mask_Ex[sq] & bb_occupied];
            bb_lance_attacks = ABB_Lance_Attacks[color, sq][ABB_Lance_Mask_Ex[color, sq] & bb_occupied];
            bb_ret |= bt.BB_Piece[color ^ 1, (int)Piece.Lance] & bb_lance_attacks;
            return bb_ret;
        }

        public static BitBoard AttacksToLongPiece(BoardTree bt, int sq, int color)
        {
            BitBoard bb_ret, bb_occupied, bb_bh, bb_rd, bb_lance_attacks;
            bb_occupied = bt.BB_Occupied[(int)Common.Color.Black] | bt.BB_Occupied[(int)Common.Color.White];
            bb_bh = bt.BB_Piece[color, (int)Piece.Bishop] | bt.BB_Piece[color, (int)Piece.Horse];
            bb_ret = bb_bh & ABB_Diagonal_Attacks[sq][ABB_Diagonal_Mask_Ex[sq] & bb_occupied];
            bb_rd = bt.BB_Piece[color, (int)Piece.Rook] | bt.BB_Piece[color, (int)Piece.Dragon];
            bb_ret |= bb_rd & ABB_Cross_Attacks[sq][ABB_Cross_Mask_Ex[sq] & bb_occupied];
            bb_lance_attacks = ABB_Lance_Attacks[color ^ 1, sq][ABB_Lance_Mask_Ex[color ^ 1, sq] & bb_occupied];
            bb_ret |= bt.BB_Piece[color, (int)Piece.Lance] & bb_lance_attacks;
            return bb_ret;
        }

        public static bool IsDiscoverKing(BoardTree bt, int ifrom, int ito, int color)
        {
            Direction idirec = Adirec[bt.SQ_King[color], ifrom];
            if (idirec != Direction.Direc_Misc && idirec != Adirec[bt.SQ_King[color], ito] && IsPinnedOnKing(bt, ifrom, idirec, color) != 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        public static bool IsDiscoverKing2(BoardTree bt, int ifrom, int ito, int color, int ipiece)
        {
            Direction idirec = Adirec[bt.SQ_King[color], ifrom];
            bt.BB_Piece[color, ipiece] ^= ABB_Mask[ifrom];
            bt.BB_Occupied[color] ^= ABB_Mask[ifrom];
            if (idirec != Direction.Direc_Misc && idirec != Adirec[bt.SQ_King[color], ito] && IsPinnedOnKing(bt, ifrom, idirec, color) != 0)
            {
                bt.BB_Piece[color, ipiece] ^= ABB_Mask[ifrom];
                bt.BB_Occupied[color] ^= ABB_Mask[ifrom];
                return true;
            }
            else
            {
                bt.BB_Piece[color, ipiece] ^= ABB_Mask[ifrom];
                bt.BB_Occupied[color] ^= ABB_Mask[ifrom];
                return false;
            }
        }
    }
}
