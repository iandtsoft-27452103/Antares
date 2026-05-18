using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BitBoard = System.UInt128;
using static Antares.Common;
using static Antares.BitOperation;
using static Antares.Board;
using static Antares.AttacksOperation;
using static Antares.Move;
using System.Net.NetworkInformation;
using System.Drawing;

namespace Antares
{
    public class Mate1Ply
    {
        public static Move MateIn1Ply(BoardTree bt, int color)
        {
            Move mate_move = new Move();
            Move null_move = new Move();
            BitBoard bb_can_escape, bb_opp_king_attacks, bb_myside_attacks, bb_enemy_attacks, bb;
            int sq, myside_attacks_count, attacks_count, sq_object;
            int[] sq_can_check_by_drop = new int[8];
            int[] sq_can_check_by_move = new int[8];
            int[] pos_array = new int[10];
            int[] pc_array = new int[10];
            int[] sq_can_escape = new int[8];
            int cnt_pos, cnt_pc;
            int cnt_d, cnt_m, cnt_e, idirec, pc, flag_promo, pos, i, j, k, index;
            bool flag;
            uint hand;
            cnt_d = cnt_m = cnt_e = 0;
            // 敵玉の位置を取得する。
            int opponent_color = color ^ 1;
            int sq_opponent_king = bt.SQ_King[opponent_color];
            bb_can_escape = BB_Full & ~bt.BB_Occupied[opponent_color];
            hand = (uint)bt.Hand[color];
            bb_opp_king_attacks = ABB_Piece_Attacks[opponent_color, (int)Piece.King, sq_opponent_king];
            while (bb_opp_king_attacks > 0)
            {
                sq = Square(bb_opp_king_attacks);
                bb_opp_king_attacks ^= ABB_Mask[sq];
                bb_myside_attacks = AttacksToPiece(bt, sq, opponent_color);
                myside_attacks_count = (int)BitBoard.PopCount(bb_myside_attacks);
                flag = false;
                if (myside_attacks_count >= 2 && bt.Board[sq] == (int)Piece.Empty)
                {
                    //If there are attacks from opponent pieces except king, opponents can capture the checker.
                    flag = true;
                }
                if ((bb_can_escape & ABB_Mask[sq]) > 0)
                {
                    // If there are attacks from your pieces, you maybe generate escape move.
                    if (IsAttacked(bt, sq, opponent_color) == 0)
                    {
                        sq_can_escape[cnt_e++] = sq;
                    }
                }
                if (bt.Board[sq] == (int)Piece.Empty && flag == false)
                {
                    sq_can_check_by_drop[cnt_d++] = sq;
                }
                bb_enemy_attacks = IsAttacked(bt, sq, color ^ 1);
                if (bt.Board[sq] != (int)Piece.Empty && (bt.BB_Occupied[opponent_color] & ABB_Mask[sq]) > 0 && bb_enemy_attacks > 0)
                {
                    sq_can_check_by_move[cnt_m++] = sq;
                }
                if (myside_attacks_count < 2 && bt.Board[sq] == (int)Piece.Empty && bb_enemy_attacks > 0)
                {
                    sq_can_check_by_move[cnt_m++] = sq;
                }
            }
            for (i = 0; i < cnt_d; i++)
            {
                sq = sq_can_check_by_drop[i];
                idirec = (int)Adirec[sq, sq_opponent_king];
                Dictionary<int, List<int>> pt = Piece_Table[opponent_color];
                bb = AttacksToPiece(bt, sq, opponent_color);
                cnt_pos = 0;
                cnt_pc = 0;
                while (bb > 0)
                {
                    pos = Square(bb);
                    bb ^= ABB_Mask[pos];
                    pos_array[cnt_pos++] = pos;
                    pc_array[cnt_pc++] = bt.Board[pos];
                }
                List<int> pcs = pt[idirec];
                if (hand > 0)
                {
                    for (j = 0; j  < pt.Count; j++)
                    {
                        pc = pcs[j];
                        if (pc > (int)Piece.Rook)
                        {
                            break;
                        }
                        if ((pc != (int)Piece.Pawn) && (hand & Hand_Mask[pc]) > 0)
                        {
                            if (cnt_e == 0)
                            {
                                mate_move.Pack(Square_NB + pc - 1, sq, (Piece)pc, 0, 0);
                                return mate_move;
                            }
                            int counter = 0;
                            bool mate_flag = true;
                            for (k = 0; k < cnt_e; k++)
                            {
                                sq_object = sq_can_escape[k];
                                if (sq == sq_object)
                                {
                                    counter++;
                                }
                                if (!IsCanEscape(bt, color, sq, pc, sq_opponent_king, sq_object, false) && !IsCanCapture(bt, color, opponent_color, sq, true, -1, pc))
                                {
                                    counter++;
                                }
                                else
                                {
                                    mate_flag = false;
                                }
                            }
                            if (counter == cnt_e && mate_flag)
                            {
                                mate_move.Pack(Square_NB + pc - 1, sq, (Piece)pc, 0, 0);
                                return mate_move;
                            }
                        }
                    }
                }
            }
            for (i = 0; i < cnt_m; i++)
            {
                sq = sq_can_check_by_move[i];
                idirec = (int)Adirec[sq, sq_opponent_king];
                Dictionary<int, List<int>> pt = Piece_Table[opponent_color];
                bb = AttacksToPiece(bt, sq, color);
                attacks_count = (int)BitBoard.PopCount(bb);
                if (attacks_count < 2 && bb > 0)
                {
                    pos = Square(bb);
                    BitBoard bb2 = AttacksToLongPiece(bt, pos, color);
                    while (bb2 > 0)
                    {
                        int sq2 = Square(bb2);
                        bb2 ^= ABB_Mask[sq2];
                        int idirec2 = (int)Adirec[sq2, sq_opponent_king];
                        if (idirec == idirec2)
                        {
                            if (cnt_e == 0)
                            {
                                if (!IsCanCapture(bt, color, opponent_color, sq, false, pos, Math.Abs(bt.Board[pos])))
                                {
                                    mate_move.Pack(pos, sq, (Piece)Math.Abs(bt.Board[pos]), (Piece)Math.Abs(bt.Board[sq]), 0);
                                    return mate_move;
                                }
                            }
                            else if (cnt_e == 1)
                            {
                                int sq3 = sq_can_escape[0];
                                int idirec3 = (int)Adirec[sq3, sq_opponent_king];
                                if (!IsCanCapture(bt, color, opponent_color, sq, false, pos, Math.Abs(bt.Board[pos])))
                                {
                                    if (Math.Abs(idirec) == Math.Abs(idirec3))
                                    {
                                        switch (Math.Abs(idirec))
                                        {
                                            case (int)Direction.Direc_File_U2d:
                                                if (Math.Abs(bt.Board[pos]) == (int)Piece.Lance || Math.Abs(bt.Board[pos]) == (int)Piece.Rook || Math.Abs(bt.Board[pos]) == (int)Piece.Dragon)
                                                {
                                                    mate_move.Pack(pos, sq, (Piece)Math.Abs(bt.Board[pos]), (Piece)Math.Abs(bt.Board[sq]), 0);
                                                    return mate_move;
                                                }
                                                break;
                                            case (int)Direction.Direc_Rank_L2r:
                                                if (Math.Abs(bt.Board[pos]) == (int)Piece.Rook || Math.Abs(bt.Board[pos]) == (int)Piece.Dragon)
                                                {
                                                    mate_move.Pack(pos, sq, (Piece)Math.Abs(bt.Board[pos]), (Piece)Math.Abs(bt.Board[sq]), 0);
                                                    return mate_move;
                                                }
                                                break;
                                            case (int)Direction.Direc_Diag1_U2d:
                                            case (int)Direction.Direc_Diag2_U2d:
                                                if (Math.Abs(bt.Board[pos]) == (int)Piece.Bishop || Math.Abs(bt.Board[pos]) == (int)Piece.Horse)
                                                {
                                                    mate_move.Pack(pos, sq, (Piece)Math.Abs(bt.Board[pos]), (Piece)Math.Abs(bt.Board[sq]), 0);
                                                    return mate_move;
                                                }
                                                break;
                                        }
                                    }

                                }
                            }
                        }
                        else
                        {
                            //bb_temp2 = ABB_Mask[sq] & BB_Color_Position[opponent_color];
                            if (cnt_e == 0 && (ABB_Piece_Attacks[color, (int)Piece.Gold, sq] & ABB_Piece_Attacks[opponent_color, (int)Piece.King, sq_opponent_king]) > 0 && (ABB_Mask[sq] & BB_Color_Position[opponent_color]) > 0)
                            {
                                bb_myside_attacks = AttacksToPiece(bt, sq, opponent_color);
                                myside_attacks_count = (int)BitBoard.PopCount(bb_myside_attacks);
                                Direction idirec3 = Adirec[pos, bt.SQ_King[color]];
                                bt.BB_Occupied[color] ^= ABB_Mask[pos];
                                bb = IsPinnedOnKing(bt, pos, idirec3, color);
                                bt.BB_Occupied[color] ^= ABB_Mask[pos];
                                if (myside_attacks_count < 2 && bb == 0)
                                {
                                    mate_move.Pack(pos, sq, (Piece)Math.Abs(bt.Board[pos]), (Piece)Math.Abs(bt.Board[sq]), 1);
                                    return mate_move;
                                }
                            }
                        }
                    }
                    continue;
                }
                cnt_pos = cnt_pc = 0;
                while (bb > 0)
                {
                    pos = Square(bb);
                    bb ^= ABB_Mask[pos];
                    pos_array[cnt_pos++] = pos;
                    pc_array[cnt_pc++] = bt.Board[pos];
                }

                List<int> pcs = pt[idirec];
                if (cnt_pos == 0)// This maybe not make sense.
                {
                    continue;
                }
                index = 0;
                while (index < cnt_pos)
                {
                    pos = pos_array[index];
                    pc = Math.Abs(pc_array[index]);
                    if (pc == (int)Piece.King)
                    {
                        index++;
                        continue;
                    }
                    idirec = (int)Adirec[pos, sq_opponent_king];
                    if (IsDiscoverKing2(bt, pos, sq, color, pc))
                    {
                        index++;
                        continue;
                    }
                    if (pcs.Contains(pc))
                    {
                        if (LongPieces2.Contains(pc))
                        {
                            if (cnt_e == 0)
                            {
                                if (LongPieces.Contains(pc) && !IsCanCapture(bt, color, opponent_color, sq, false, pos, pc))
                                {
                                    if ((ABB_Mask[sq] & BB_Color_Position[opponent_color]) > 0)
                                    {
                                        flag_promo = 1;
                                    }
                                    else
                                    {
                                        flag_promo = 0;
                                    }
                                    mate_move.Pack(pos, sq, (Piece)pc, (Piece)Math.Abs(bt.Board[sq]), flag_promo);
                                    return mate_move;
                                }
                            }
                            flag = false;
                            for (j = 0; j < cnt_e; j++)
                            {
                                sq_object = sq_can_escape[j];
                                if (sq == sq_object)
                                {
                                    continue;
                                }
                                if (!IsCanEscape(bt, color, sq, pc, sq_opponent_king, sq_object, false) && !IsCanCapture(bt, color, opponent_color, sq, false, pos, pc))
                                {
                                    if ((ABB_Mask[pos] & BB_Color_Position[opponent_color]) > 0 || (ABB_Mask[sq] & BB_Color_Position[opponent_color]) > 0)
                                    {
                                        flag_promo = 1;
                                    }
                                    else
                                    {
                                        flag_promo = 0;
                                    }
                                    mate_move.Pack(pos, sq, (Piece)pc, (Piece)Math.Abs(bt.Board[sq]), flag_promo);
                                    flag = true;
                                    //return mate_move;
                                }
                                else
                                {
                                    flag = false;
                                    mate_move = new Move();
                                    break;
                                }
                            }
                            if (flag && mate_move.Value != 0)
                            {
                                return mate_move;
                            }
                        }
                        else if (pc == (int)Piece.Dragon || pc == (int)Piece.Horse)
                        {
                            if (cnt_e == 0)
                            {
                                if (!IsCanCapture(bt, color, opponent_color, sq, false, pos, pc))
                                {
                                    mate_move.Pack(pos, sq, (Piece)pc, (Piece)Math.Abs(bt.Board[sq]), 0);
                                    return mate_move;
                                }
                            }
                            flag = false;
                            for (j = 0; j < cnt_e; j++)
                            {
                                sq_object = sq_can_escape[j];
                                if (sq == sq_object)
                                {
                                    continue;
                                }
                                if (!IsCanEscape(bt, color, sq, pc, sq_opponent_king, sq_object, false) && !IsCanCapture(bt, color, opponent_color, sq, false, pos, pc))
                                {
                                    mate_move.Pack(pos, sq, (Piece)pc, (Piece)Math.Abs(bt.Board[sq]), 0);
                                    flag = true;
                                }
                                else
                                {
                                    flag = false;
                                    mate_move = new Move();
                                    break;
                                }
                            }
                            if (flag && mate_move.Value != 0)
                            {
                                return mate_move;
                            }
                        }
                        else
                        {
                            switch (pc)
                            {
                                case (int)Piece.Gold:
                                case (int)Piece.Pro_Pawn:
                                case (int)Piece.Pro_Lance:
                                case (int)Piece.Pro_Knight:
                                case (int)Piece.Pro_Silver:
                                case (int)Piece.Silver:
                                    if (cnt_e == 0)
                                    {
                                        if (!IsCanCapture(bt, color, opponent_color, sq, false, pos, pc))
                                        {
                                            mate_move.Pack(pos, sq, (Piece)pc, (Piece)Math.Abs(bt.Board[sq]), 0);
                                            return mate_move;
                                        }
                                    }
                                    flag = false;
                                    for (j = 0; j < cnt_e; j++)
                                    {
                                        sq_object = sq_can_escape[j];
                                        if (sq == sq_object)
                                        {
                                            continue;
                                        }
                                        if (!IsCanEscape(bt, color, sq, pc, sq_opponent_king, sq_object, false) && !IsCanCapture(bt, color, opponent_color, sq, false, pos, pc))
                                        {
                                            mate_move.Pack(pos, sq, (Piece)pc, (Piece)Math.Abs(bt.Board[sq]), 0);
                                            flag = true;// テストしていない。
                                        }
                                        else
                                        {
                                            flag = false;
                                            mate_move = new Move();
                                            break;
                                        }
                                    }
                                    if (flag && mate_move.Value != 0)
                                    {
                                        return mate_move;
                                    }
                                    break;
                            }
                        }

                        // ここでは成った方が得な場合でも香不成、歩不成で詰ます。
                        // 複雑な心境だが、flag_promoの判定を入れると少し遅くなる。
                        //mate_move.Pack(pos, sq, pc, abs(bt.board[sq]), 0);
                        //return mate_move;
                    }

                    if (pc > (int)Piece.Rook)
                    {
                        index++;
                        continue;
                    }

                    int pc_promote = pc + Promote;
                    // knight promote move
                    // Knight cannnot mate opponent king from neighbour 8 Square.
                    if ((pcs.Contains(pc_promote) && pc == (int)Piece.Knight && (BB_Rev_Color_Position[color] & ABB_Mask[sq]) > 0))
                    {
                        if (cnt_e == 0)
                        {
                            if (!IsCanCapture(bt, color, opponent_color, sq, false, pos, pc) && (ABB_Piece_Attacks[color, (int)Piece.Gold, sq] & ABB_Mask[sq_opponent_king]) > 0)
                            {
                                mate_move.Pack(pos, sq, (Piece)pc, (Piece)Math.Abs(bt.Board[sq]), 1);
                                return mate_move;
                            }
                        }
                        flag = false;
                        for (j = 0; j < cnt_e; j++)
                        {
                            sq_object = sq_can_escape[j];
                            if (sq == sq_object)
                            {
                                continue;
                            }
                            if (!IsCanEscape(bt, color, sq, pc, sq_opponent_king, sq_object, true) && !IsCanCapture(bt, color, opponent_color, sq, false, pos, pc))
                            {
                                flag = true;
                                mate_move.Pack(pos, sq, (Piece)pc, (Piece)Math.Abs(bt.Board[sq]), 1);
                                //return mate_move;
                            }
                            else
                            {
                                flag = false;
                                mate_move = new Move();
                                break;
                            }
                        }
                        if (flag && mate_move.Value != 0)
                        {
                            return mate_move;
                        }
                    }
                    // lance promote move or pawn promote move
                    if (pcs.Contains(pc_promote) && ShortPieces.Contains(pc) && (BB_Rev_Color_Position[color] & ABB_Mask[sq]) > 0)
                    {
                        if (cnt_e == 0)
                        {
                            if (!IsCanCapture(bt, color, opponent_color, sq, false, pos, pc) && (ABB_Piece_Attacks[color, (int)Piece.Gold, sq] & ABB_Mask[sq_opponent_king]) > 0)
                            {
                                mate_move.Pack(pos, sq, (Piece)pc, (Piece)Math.Abs(bt.Board[sq]), 1);
                                return mate_move;
                            }
                        }
                        flag = false;
                        for (j = 0; j < cnt_e; j++)
                        {
                            sq_object = sq_can_escape[j];
                            if (sq == sq_object)
                            {
                                continue;
                            }
                            if (!IsCanEscape(bt, color, sq, pc, sq_opponent_king, sq_object, true) && !IsCanCapture(bt, color, opponent_color, sq, false, pos, pc))
                            {
                                flag = true;
                                mate_move.Pack(pos, sq, (Piece)pc, (Piece)Math.Abs(bt.Board[sq]), 1);
                                //return mate_move;
                            }
                            else
                            {
                                flag = false;
                                mate_move = new Move();
                                break;
                            }
                        }
                        if (flag && mate_move.Value != 0)
                        {
                            mate_move.Pack(pos, sq, (Piece)pc, (Piece)Math.Abs(bt.Board[sq]), 1);
                            return mate_move;
                        }
                    }
                    // silver promote move
                    if (pc == (int)Piece.Silver)
                    {
                        if (pcs.Contains(pc_promote) && (BB_Rev_Color_Position[color] & ABB_Mask[sq]) > 0 || (BB_Rev_Color_Position[color] & ABB_Mask[pos]) > 0)
                        {
                            if (cnt_e == 0)
                            {
                                if (!IsCanCapture(bt, color, opponent_color, sq, false, pos, pc) && (ABB_Piece_Attacks[color, (int)Piece.Gold, sq] & ABB_Mask[sq_opponent_king]) > 0)
                                {
                                    mate_move.Pack(pos, sq, (Piece)pc, (Piece)Math.Abs(bt.Board[sq]), 1);
                                    return mate_move;
                                }
                            }
                            flag = false;
                            for (j = 0; j < cnt_e; j++)
                            {
                                sq_object = sq_can_escape[j];
                                if (sq == sq_object)
                                {
                                    continue;
                                }
                                if (!IsCanEscape(bt, color, sq, pc, sq_opponent_king, sq_object, true) && !IsCanCapture(bt, color, opponent_color, sq, false, pos, pc))
                                {
                                    mate_move.Pack(pos, sq, (Piece)pc, (Piece)Math.Abs(bt.Board[sq]), 1);
                                    flag = true;
                                    //return mate_move;
                                }
                                else
                                {
                                    flag = false;
                                    mate_move = new Move();
                                    break;
                                }
                            }
                            if (flag && mate_move.Value != 0)
                            {
                                return mate_move;
                            }
                        }

                    }

                    if (pc < (int)Piece.Bishop)
                    {
                        index++;
                        continue;
                    }

                    // rook promote move or bishop promote move
                    if (pcs.Contains(pc_promote) && LongPieces.Contains(pc) && (BB_Rev_Color_Position[color] & ABB_Mask[sq]) > 0 || (BB_Rev_Color_Position[color] & ABB_Mask[pos]) > 0)
                    {
                        if (cnt_e == 0)
                        {
                            if (!IsCanCapture(bt, color, opponent_color, sq, false, pos, pc) && (ABB_Piece_Attacks[color, (int)Piece.King, sq] & ABB_Mask[sq_opponent_king]) > 0)
                            {
                                // ※ここは修正後未テスト
                                if ((ABB_Mask[pos] & BB_Color_Position[opponent_color]) > 0 || (ABB_Mask[sq] & BB_Color_Position[opponent_color]) > 0 /*(ABB_Mask[pos] & BB_Color_Position[opponent_color]) > 0*/)
                                {
                                    flag_promo = 1;
                                }
                                else
                                {
                                    flag_promo = 0;// この場合は成らないか？
                                }
                                mate_move.Pack(pos, sq, (Piece)pc, (Piece)Math.Abs(bt.Board[sq]), flag_promo);
                                return mate_move;
                            }
                        }
                        flag = false;
                        for (j = 0; j < cnt_e; j++)
                        {
                            sq_object = sq_can_escape[j];
                            if (sq == sq_object)
                            {
                                continue;
                            }
                            if (!IsCanEscape(bt, color, sq, pc, sq_opponent_king, sq_object, true) && !IsCanCapture(bt, color, opponent_color, sq, false, pos, pc))
                            {
                                if ((ABB_Mask[pos] & BB_Color_Position[opponent_color]) > 0 || (ABB_Mask[sq] & BB_Color_Position[opponent_color]) > 0 )
                                {
                                    flag_promo = 1;
                                }
                                else
                                {
                                    flag_promo = 0;// この場合は成らないか？
                                }
                                mate_move.Pack(pos, sq, (Piece)pc, (Piece)Math.Abs(bt.Board[sq]), flag_promo);
                                flag = true;
                            }
                            else
                            {
                                flag = false;
                                mate_move = new Move();
                                break;
                            }
                        }
                        if (flag && mate_move.Value != 0)
                        {
                            return mate_move;
                        }
                    }
                    index++;
                }
            }
            // You cannot mate opponnent king from neighbour 8 square.
            // You maybe mate opponnent move using knight.
            pc = (int)Piece.Knight;
            BitBoard bb_occupied, bb_opponent_attacks_to_sq, bb_my_knight_attacks;
            bb_occupied = bt.BB_Occupied[(int)Common.Color.Black] | bt.BB_Occupied[(int)Common.Color.White];
            bb = ABB_Piece_Attacks[opponent_color, pc, sq_opponent_king] & ((~bb_occupied & BB_Full) | bt.BB_Occupied[opponent_color]);
            while (bb > 0)
            {
                sq = Square(bb);
                bb ^= ABB_Mask[sq];
                bb_opponent_attacks_to_sq = AttacksToPiece(bt, sq, opponent_color);
                if ((hand & Hand_Mask[pc]) > 0 && bt.Board[sq] == (int)Piece.Empty && cnt_e == 0 && bb_opponent_attacks_to_sq == 0)
                {
                    // drop knight
                    mate_move.Pack(Square_NB + pc - 1, sq, (Piece)pc, 0, 0);
                    return mate_move;
                }
                bb_my_knight_attacks = ABB_Piece_Attacks[opponent_color, pc, sq] & bt.BB_Piece[color, (int)Piece.Knight];
                if (bb_my_knight_attacks > 0 && cnt_e == 0 && bb_opponent_attacks_to_sq == 0)
                {
                    pos = Square(bb_my_knight_attacks);
                    bb_my_knight_attacks ^= ABB_Mask[pos];
                    if (IsDiscoverKing2(bt, pos, sq, color, pc))
                    {
                        continue;
                    }
                    mate_move.Pack(pos, sq, (Piece)pc, (Piece)Math.Abs(bt.Board[sq]), 0);
                }
            }
            if (mate_move.Value != 0)
                return mate_move;
            return null_move;
        }

        public static bool IsCanEscape(BoardTree bt, int color, int sq_checker, int pc_checker, int sq_opponent_king, int sq_object, bool is_promo)
        {
            BitBoard bb_occupied, bb_attacks;
            bb_occupied = bt.BB_Occupied[(int)Common.Color.Black] | bt.BB_Occupied[(int)Common.Color.White];
            bb_occupied ^= (ABB_Mask[sq_opponent_king] | ABB_Mask[sq_object]);
            bb_attacks = 0;
            switch (pc_checker)
            {
                case (int)Piece.Rook:
                    bb_attacks = ABB_Cross_Attacks[sq_checker][ABB_Cross_Mask_Ex[sq_checker] & bb_occupied];
                    break;
                case (int)Piece.Dragon:
                    bb_attacks = ABB_Cross_Attacks[sq_checker][ABB_Cross_Mask_Ex[sq_checker] & bb_occupied];
                    bb_attacks |= ABB_Piece_Attacks[color, (int)Piece.King, sq_checker];
                    break;
                case (int)Piece.Bishop:
                    bb_attacks = ABB_Diagonal_Attacks[sq_checker][ABB_Diagonal_Mask_Ex[sq_checker] & bb_occupied];
                    break;
                case (int)Piece.Horse:
                    bb_attacks = ABB_Diagonal_Attacks[sq_checker][ABB_Diagonal_Mask_Ex[sq_checker] & bb_occupied];
                    bb_attacks |= ABB_Piece_Attacks[color, (int)Piece.King, sq_checker];
                    break;
                case (int)Piece.Pawn:
                case (int)Piece.Knight:
                case (int)Piece.Silver:
                    if (is_promo)
                    {
                        bb_attacks = ABB_Piece_Attacks[color, (int)Piece.Gold, sq_checker];
                    }
                    else
                    {
                        bb_attacks = ABB_Piece_Attacks[color, pc_checker, sq_checker];
                    }
                    break;
                case (int)Piece.Lance:
                    if (is_promo)
                    {
                        bb_attacks = ABB_Piece_Attacks[color, (int)Piece.Gold, sq_checker];
                    }
                    else
                    {
                        bb_attacks = ABB_Lance_Attacks[color, sq_checker][ABB_Lance_Mask_Ex[color, sq_checker] & bb_occupied];
                    }
                    break;
                default:
                    bb_attacks = ABB_Piece_Attacks[color, pc_checker, sq_checker];
                    break;
            }
            bb_attacks &= ABB_Mask[sq_object];
            if (bb_attacks > 0)
            {
                return false;
            }
            return true;
        }

        public static bool IsCanCapture(BoardTree bt, int color, int opponent_color, int sq_object, bool is_drop, int ifrom, int ipiece)
        {
            BitBoard bb_myside_attacks, bb_opp_attacks, bb, bb2, bb3;
            //AttacksOperation atkop;
            int myside_attacks_count, opp_attacks_count;
            int idirec;
            bb_myside_attacks = AttacksToPiece(bt, sq_object, color);
            myside_attacks_count = (int)BitBoard.PopCount(bb_myside_attacks);
            bb_opp_attacks = AttacksToPiece(bt, sq_object, opponent_color);
            opp_attacks_count = (int)BitBoard.PopCount(bb_opp_attacks);
            if (opp_attacks_count > 1)
                return true;
            if ((opp_attacks_count == 1) && (myside_attacks_count == 0))
            {
                //敵玉の利きのみだが、味方の駒が対象マスに利いていない場合
                return true;
            }
            if (opp_attacks_count >= myside_attacks_count)
            {
                if ((opp_attacks_count == myside_attacks_count) && is_drop)
                {
                    // 敵玉の利きのみで、味方の駒の利きはひとつだが、駒打ち王手の場合
                    return false;
                }
                if (is_drop)
                    return true;
                bt.BB_Occupied[color] ^= ABB_Mask[ifrom];
                bt.BB_Piece[color, ipiece] ^= ABB_Mask[ifrom];
                bb = IsAttacked(bt, bt.SQ_King[opponent_color], color);
                bb2 = IsAttacked(bt, bt.SQ_King[color], color);
                bb3 = 0;
                switch (ipiece)
                {
                    case (int)Piece.Pawn:
                    case (int)Piece.Lance:
                    case (int)Piece.Rook:
                    case (int)Piece.Dragon:
                        idirec = (int)Adirec[ifrom, sq_object];
                        if (Math.Abs(idirec) == (int)Direction.Direc_File_U2d)
                        {
                            bb3 = IsAttacked(bt, sq_object, color);
                        }
                        break;
                }
                bt.BB_Occupied[color] ^= ABB_Mask[ifrom];
                bt.BB_Piece[color, ipiece] ^= ABB_Mask[ifrom];
                if (bb2 > 0)
                    return true;
                if (bb > 0 || bb3 > 0)
                    return false;
                return true;
            }
            return false;
        }
    }
}
