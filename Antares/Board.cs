using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Net.NetworkInformation;
using System.Runtime.InteropServices.Marshalling;
using System.Security.Policy;
using System.Text;
using System.Threading.Tasks;
using static Antares.Common;
using static Antares.Hash;
using BitBoard = System.UInt128;
using Rand = System.UInt128;

namespace Antares
{
    public class Board
    {
        public static int[] MaterialBoard = { 0, 87, 232, 257, 399, 474, 619, 692, 15000, 564, 519, 540, 525, 0, 877, 995 };
        public static int[] MaterialHand = { 0, 87, 232, 257, 399, 474, 619, 692, 15000, 87, 232, 257, 399, 0, 619, 692 };

        public struct BoardTree
        {
            public BitBoard[,] BB_Piece;
            public BitBoard[] BB_Occupied;
            public int[] SQ_King;
            public int[] Hand;
            public int[] Board;
            public Common.Color RootColor;
            public Move[] RootMoves;
            public Rand[] Hash;
            public Rand CurrentHash;
            public Rand PrevHash;
            public int ply;
            public int[] EvalArray;
        }

        public static void Init(ref BoardTree BTree)
        {
            int i, j;
            BTree.BB_Occupied[0] = (BitBoard)(511 << 18) | ABB_Mask[64] | ABB_Mask[70] | (BitBoard)511;
            BTree.BB_Occupied[1] = ((BitBoard)134022655 << 54);
            BTree.BB_Piece[0, (int)Piece.Pawn] = (BitBoard)(511 << 18);
            BTree.BB_Piece[1, (int)Piece.Pawn] = (BitBoard)133955584 << 36;
            BTree.BB_Piece[0, (int)Piece.Lance] = ABB_Mask[72] | ABB_Mask[80];
            BTree.BB_Piece[1, (int)Piece.Lance] = ABB_Mask[0] | ABB_Mask[8];
            BTree.BB_Piece[0, (int)Piece.Knight] = ABB_Mask[73] | ABB_Mask[79];
            BTree.BB_Piece[1, (int)Piece.Knight] = ABB_Mask[1] | ABB_Mask[7];
            BTree.BB_Piece[0, (int)Piece.Silver] = ABB_Mask[74] | ABB_Mask[78];
            BTree.BB_Piece[1, (int)Piece.Silver] = ABB_Mask[2] | ABB_Mask[6];
            BTree.BB_Piece[0, (int)Piece.Gold] = ABB_Mask[75] | ABB_Mask[77];
            BTree.BB_Piece[1, (int)Piece.Gold] = ABB_Mask[3] | ABB_Mask[5];
            BTree.BB_Piece[0, (int)Piece.Bishop] = ABB_Mask[64];
            BTree.BB_Piece[1, (int)Piece.Bishop] = ABB_Mask[16];
            BTree.BB_Piece[0, (int)Piece.Rook] = ABB_Mask[70];
            BTree.BB_Piece[1, (int)Piece.Rook] = ABB_Mask[10];
            BTree.BB_Piece[0, (int)Piece.King] = ABB_Mask[76];
            BTree.BB_Piece[1, (int)Piece.King] = ABB_Mask[4];
            for (i = (int)Common.Color.Black; i < Color_NB; i++)
            {
                for (j = (int)Piece.Pro_Pawn; j < Piece_NB; j++)
                {
                    // 金は成らないが、そのままループを回している
                    BTree.BB_Piece[i, j] = 0;
                }
            }
            BTree.Board[0] = BTree.Board[8] = -(int)Piece.Lance;
            BTree.Board[1] = BTree.Board[7] = -(int)Piece.Knight;
            BTree.Board[2] = BTree.Board[6] = -(int)Piece.Silver;
            BTree.Board[3] = BTree.Board[5] = -(int)Piece.Gold;
            BTree.Board[4] = -(int)Piece.King;
            BTree.Board[10] = -(int)Piece.Rook;
            BTree.Board[16] = -(int)Piece.Bishop;
            BTree.Board[18] = BTree.Board[19] = BTree.Board[20] = BTree.Board[21] = BTree.Board[22] = BTree.Board[23] = BTree.Board[24] = BTree.Board[25] = BTree.Board[26] = -(int)Piece.Pawn;
            BTree.Board[72] = BTree.Board[80] = (int)Piece.Lance;
            BTree.Board[73] = BTree.Board[79] = (int)Piece.Knight;
            BTree.Board[74] = BTree.Board[78] = (int)Piece.Silver;
            BTree.Board[75] = BTree.Board[77] = (int)Piece.Gold;
            BTree.Board[76] = (int)Piece.King;
            BTree.Board[70] = (int)Piece.Rook;
            BTree.Board[64] = (int)Piece.Bishop;
            BTree.Board[54] = BTree.Board[55] = BTree.Board[56] = BTree.Board[57] = BTree.Board[58] = BTree.Board[59] = BTree.Board[60] = BTree.Board[61] = BTree.Board[62] = (int)Piece.Pawn;
            BTree.Hand[0] = 0;
            BTree.Hand[1] = 0;
            BTree.CurrentHash = HashFunc(BTree);
            BTree.RootColor = Common.Color.Black;
            BTree.SQ_King[0] = 76;
            BTree.SQ_King[1] = 4;
            BTree.ply = 1;
            BTree.PrevHash = 0;
            BTree.Hash[1] = BTree.CurrentHash;
            BTree.EvalArray = new int[Ply_Max];
        }

        public static void Clear(ref BoardTree BTree)
        {
            int i, j;
            BTree.BB_Occupied[0] = 0;
            BTree.BB_Occupied[1] = 0;
            for (i = (int)Common.Color.Black; i < Color_NB; i++)
            {
                for (j = (int)Piece.Pawn; j < Piece_NB; j++)
                {
                    BTree.BB_Piece[i, j] = 0;
                }
            }
            BTree.Hand[0] = BTree.Hand[1] = 0;
            BTree.CurrentHash = HashFunc(BTree);
            BTree.RootColor = Common.Color.Black;
            BTree.SQ_King[(int)Common.Color.Black] = 0;
            BTree.SQ_King[(int)Common.Color.White] = 0;
            BTree.ply = 1;
            BTree.PrevHash = 0;
            BTree.EvalArray = new int[Ply_Max];
        }

        public static void BoardTreeAlloc(ref BoardTree BTree)
        {
            BTree.BB_Piece = new BitBoard[Color_NB, Piece_NB];
            BTree.BB_Occupied = new BitBoard[Color_NB];
            BTree.SQ_King = new int[Color_NB];
            BTree.Hand = new int[Color_NB];
            BTree.Board = new int[Square_NB];
            //BTree.RootColor = Color.Type.Black;
            // RootMovesは初期化しない
            BTree.Hash = new Rand[Ply_Max + 1];
            BTree.ply = 1;

            for (int i = 0; i < Color_NB; i++)
            {
                BTree.BB_Occupied[i] = new BitBoard();
                BTree.SQ_King[i] = new int();
                for (int j = 0; j < Piece_NB; j++)
                {
                    BTree.BB_Piece[i, j] = new BitBoard();
                }
                BTree.Hand[i] = new int();
            }
            BTree.EvalArray = new int[Ply_Max];
        }

        public static BoardTree DeepCopy(BoardTree bt, bool flag)
        {
            BoardTree bt_base = new BoardTree();
            BoardTreeAlloc(ref bt_base);
            for (int i = 0; i < Color_NB; i++)
            {
                bt_base.BB_Occupied[i] = bt.BB_Occupied[i];
                bt_base.SQ_King[i] = bt.SQ_King[i];
                for (int j = 0; j < Piece_NB; j++)
                {
                    bt_base.BB_Piece[i, j] = bt.BB_Piece[i, j];
                }
                bt_base.Hand[i] = bt.Hand[i];
                //bt_base.Hand[0] = bt.Hand[0];
                //bt_base.Hand[1] = bt.Hand[1];
            }

            bt_base.RootColor = bt.RootColor;
            bt_base.ply = bt.ply;
            bt_base.CurrentHash = bt.CurrentHash;
            bt_base.PrevHash = bt.PrevHash;

            for (int i = 0; i < Square_NB; i++)
            {
                bt_base.Board[i] = bt.Board[i];
            }

            for (int i = 0; i < Ply_Max + 1; i++)
            {
                if (i != 0 && bt.Hash[i] == 0) { break; }
                bt_base.Hash[i] = bt.Hash[i];
                bt_base.EvalArray[i] = bt.EvalArray[i];
            }

            if (flag)
            {
                bt_base.RootMoves = new Move[bt.RootMoves.Length];
                for (int i = 0; i < bt.RootMoves.Length; i++)
                {
                    Move move = new Move();
                    move.Pack(bt.RootMoves[i].From, bt.RootMoves[i].To, bt.RootMoves[i].PieceType, bt.RootMoves[i].CapPiece, bt.RootMoves[i].FlagPromo);
                    bt_base.RootMoves[i] = move;
                }
            }
            // ※ eval_arrayのコピー処理は入っていない。
            return bt_base;
        }

        public static void Do(ref BoardTree bt, Move m, int color)
        {
            int ifrom, ito, ipiece, is_promote, icap_piece, index;
            BitBoard bb_set_clear;
            bt.PrevHash = bt.CurrentHash;
            ifrom = m.From;
            ito = m.To;
            ipiece = (int)m.PieceType;
            is_promote = m.FlagPromo;
            if (ifrom >= Square_NB)
            {
                bt.BB_Piece[color, ipiece] ^= ABB_Mask[ito];
                bt.CurrentHash ^= PieceRand[color, ipiece, ito];
                bt.Hand[color] -= Hand_Hash[ipiece];
                bt.Board[ito] = -Sign_Table[color] * ipiece;
                bt.BB_Occupied[color] ^= ABB_Mask[ito];
            }
            else
            {
                bb_set_clear = ABB_Mask[ifrom] | ABB_Mask[ito];
                //bt.BB_Occupied[color] ^= (ABB_Mask[ifrom] | ABB_Mask[ito]);
                bt.BB_Occupied[color] ^= bb_set_clear;
                bt.Board[ifrom] = (int)Piece.Empty;
                if (is_promote > 0)
                {
                    bt.BB_Piece[color, ipiece] ^= ABB_Mask[ifrom];
                    bt.BB_Piece[color, ipiece + Promote] ^= ABB_Mask[ito];
                    bt.CurrentHash ^= PieceRand[color, ipiece, ifrom] ^ PieceRand[color, ipiece + Promote, ito];
                    bt.Board[ito] = -Sign_Table[color] * (ipiece + Promote);
                }
                else
                {
                    if (ipiece == (int)Piece.King)
                    {
                        bt.SQ_King[color] = ito;
                    }
                    bt.BB_Piece[color, ipiece] ^= bb_set_clear;
                    bt.CurrentHash ^= PieceRand[color, ipiece, ifrom] ^ PieceRand[color, ipiece, ito];
                    bt.Board[ito] = -Sign_Table[color] * ipiece;
                }
                icap_piece = index = (int)m.CapPiece;
                if (icap_piece > 0)
                {
                    if (icap_piece > (int)Piece.King)
                    {
                        index -= Promote;
                    }
                    bt.Hand[color] += Hand_Hash[index];
                    bt.BB_Piece[color ^ 1, icap_piece] ^= ABB_Mask[ito];
                    bt.CurrentHash ^= PieceRand[color ^ 1, icap_piece, ito];
                    bt.BB_Occupied[color ^ 1] ^= ABB_Mask[ito];
                }
            }
            bt.Hash[bt.ply] = bt.PrevHash;
            bt.Hash[bt.ply + 1] = bt.CurrentHash;
            bt.ply += 1;
        }

        public static void UnDo(ref BoardTree bt, Move m, int color)
        {
            int ifrom, ito, ipiece, is_promote, icap_piece, index;
            BitBoard bb_set_clear;
            bt.CurrentHash = bt.PrevHash;
            ifrom = m.From;
            ito = m.To;
            ipiece = (int)m.PieceType;
            is_promote = m.FlagPromo;
            if (ifrom >= Square_NB)
            {
                bt.BB_Piece[color, ipiece] ^= ABB_Mask[ito];
                bt.Hand[color] += Hand_Hash[ipiece];
                bt.Board[ito] = (int)Piece.Empty;
                bt.BB_Occupied[color] ^= ABB_Mask[ito];
            }
            else
            {
                bb_set_clear = ABB_Mask[ifrom] | ABB_Mask[ito];
                bt.BB_Occupied[color] ^= bb_set_clear;
                bt.Board[ifrom] = -Sign_Table[color] * ipiece;
                if (is_promote > 0)
                {
                    bt.BB_Piece[color, ipiece] ^= ABB_Mask[ifrom];
                    bt.BB_Piece[color, ipiece + Promote] ^= ABB_Mask[ito];
                }
                else
                {
                    if (ipiece == (int)Piece.King)
                    {
                        bt.SQ_King[color] = ifrom;
                    }
                    bt.BB_Piece[color, ipiece] ^= bb_set_clear;
                }
                icap_piece = index = (int)m.CapPiece;
                if (icap_piece > 0)
                {
                    if (icap_piece > (int)Piece.King)
                    {
                        index -= Promote;
                    }
                    bt.Hand[color] -= Hand_Hash[index];
                    bt.BB_Piece[color ^ 1, icap_piece] ^= ABB_Mask[ito];
                    bt.BB_Occupied[color ^ 1] ^= ABB_Mask[ito];
                    bt.Board[ito] = Sign_Table[color] * icap_piece;
                }
                else
                {
                    bt.Board[ito] = (int)Piece.Empty;
                }
            }
            bt.PrevHash = bt.Hash[bt.ply - 2];
            bt.Hash[bt.ply] = 0; //配列のインデックスが1ずれていないかチェックする
            bt.ply -= 1;
        }

        public static void DoNull(ref BoardTree bt)
        {
            bt.Hash[bt.ply + 1] = bt.CurrentHash;
            bt.ply += 1;
        }

        public static void UnDoNull(ref BoardTree bt)
        {
            bt.Hash[bt.ply] = 0;
            bt.ply -= 1;
        }

        // 戻り値
        // 0: 宣言勝ちの局面ではない。
        // 1: 先手の勝ち
        // 2: 後手の勝ち
        public static int IsDeclarationWin(BoardTree bt)
        {
            BitBoard bb0, bb1, bb_object, bb_temp;
            int i, black_score, white_score;
            black_score = white_score = 0;
            int b_tekijin_piece_count, w_tekijin_piece_count;
            b_tekijin_piece_count = w_tekijin_piece_count = 0;
            int[] b_hand_piece_count = new int[(int)Piece.Rook + 1];
            int[] w_hand_piece_count = new int[(int)Piece.Rook + 1];
            int[] b_board_piece_count = new int[Piece_NB];
            int[] w_board_piece_count = new int[Piece_NB];
            bb0 = bt.BB_Piece[(int)Common.Color.Black, (int)Piece.King] & BB_White_Position;
            bb1 = bt.BB_Piece[(int)Common.Color.White, (int)Piece.King] & BB_Black_Position;
            if (bb0 == 0 && bb1 == 0)
            {
                return 0;
            }
            if (bb0 > 0)
            {
                for (i = (int)Piece.Pawn; i <= (int)Piece.Rook; i++)
                {
                    b_hand_piece_count[i] = (bt.Hand[(int)Common.Color.Black] & Hand_Mask[i]) >> Hand_Rev_Bit[i];
                    if (i >= (int)Piece.Bishop)
                    {
                        black_score += 5 * b_hand_piece_count[i];
                    }
                    else
                    {
                        black_score += b_hand_piece_count[i];
                    }
                }
                for (i = (int)Piece.Pawn; i <= (int)Piece.Dragon; i++)
                {
                    if (i == (int)Piece.None)
                        continue;
                    bb_object = bt.BB_Piece[(int)Common.Color.Black, i] & BB_Rev_Color_Position[(int)Common.Color.Black];
                    b_board_piece_count[i] = (int)BitBoard.PopCount(bb_object);
                    b_tekijin_piece_count += b_board_piece_count[i];
                    bb_temp = BB_DMZ | BB_Rev_Color_Position[(int)Common.Color.White];
                    bb_object = bb_temp & bt.BB_Piece[(int)Common.Color.Black, i];
                    b_board_piece_count[i] += (int)BitBoard.PopCount(bb_object);
                    if (i == (int)Piece.King)
                        continue;
                    if (i == (int)Piece.Bishop || i == (int)Piece.Rook || i >= (int)Piece.Horse)
                    {
                        black_score += 5 * b_board_piece_count[i];
                    }
                    else
                    {
                        black_score += b_board_piece_count[i];
                    }
                }
            }
            if (bb1 > 0)
            {
                for (i = (int)Piece.Pawn; i <= (int)Piece.Rook; i++)
                {
                    w_hand_piece_count[i] = (bt.Hand[(int)Common.Color.White] & Hand_Mask[i]) >> Hand_Rev_Bit[i];
                    if (i >= (int)Piece.Bishop)
                    {
                        white_score += 5 * w_hand_piece_count[i];
                    }
                    else
                    {
                        white_score += w_hand_piece_count[i];
                    }
                }
                for (i = (int)Piece.Pawn; i <= (int)Piece.Dragon; i++)
                {
                    if (i == (int)Piece.None)
                        continue;
                    bb_object = bt.BB_Piece[(int)Common.Color.White, i] & BB_Rev_Color_Position[(int)Common.Color.White];
                    w_board_piece_count[i] = (int)BitBoard.PopCount(bb_object);
                    w_tekijin_piece_count += w_board_piece_count[i];
                    bb_temp = BB_DMZ | BB_Rev_Color_Position[(int)Common.Color.Black];
                    bb_object = bb_temp & bt.BB_Piece[(int)Common.Color.White, i];
                    w_board_piece_count[i] += (int)BitBoard.PopCount(bb_object);
                    if (i == (int)Piece.King)
                        continue;
                    if (i == (int)Piece.Bishop || i == (int)Piece.Rook || i >= (int)Piece.Horse)
                    {
                        white_score += 5 * w_board_piece_count[i];
                    }
                    else
                    {
                        white_score += w_board_piece_count[i];
                    }
                }
            }
            if (bb0 > 0 && black_score >= 28 && b_tekijin_piece_count >= 10)
                return 1;
            if (bb1 > 0 && white_score >= 27 && w_tekijin_piece_count >= 10)
                return 2;
            return 0;
        }

        public static int IsRepetition(BoardTree bt, TT tt)
        {
            int i, counter;
            int limit = bt.ply - 12;
            if (limit < 1)
                return 0;
            counter = 0;
            i = bt.ply;
            while (i >= limit)
            {
                if (bt.CurrentHash == bt.Hash[i])
                    counter++;
                i--;
            }
            // 手抜きのため、同一局面3回で千日手と判定する。
            if (counter > 2)
            {
                if (tt.is_check.ContainsKey(bt.CurrentHash))
                {
                    bool b = tt.is_check[bt.CurrentHash];
                    if (!b)
                    {
                        return 1;// 千日手
                    }
                    else
                    {
                        return 2;// 連続王手の千日手
                    }
                }
            }
            return 0;
        }

        public static short CalcMaterial(BoardTree BTree)
        {
            const int FV_SCALE = 32;
            int material = 0;
            for (int i = (int)Piece.Pawn; i <= (int)Piece.Rook; i++)
            {
                material += ((BTree.Hand[0] & Hand_Mask[i]) >> Hand_Rev_Bit[i]) * MaterialHand[i];
            }
            for (int i = (int)Piece.Pawn; i <= (int)Piece.Rook; i++)
            {
                material -= ((BTree.Hand[1] & Hand_Mask[i]) >> Hand_Rev_Bit[i]) * MaterialHand[i];
            }
            for (int i = (int)Piece.Pawn; i < Piece_NB; i++)
            {
                if (i == (int)Piece.King) continue;
                if (i == (int)Piece.None) continue;
                BitBoard bb = BTree.BB_Piece[0, i];
                material += (int)BitBoard.PopCount(bb) * MaterialBoard[i];
            }
            for (int i = (int)Piece.Pawn; i < Piece_NB; i++)
            {
                if (i == (int)Piece.King) continue;
                if (i == (int)Piece.None) continue;
                BitBoard bb = BTree.BB_Piece[1, i];
                material -= (int)BitBoard.PopCount(bb) * MaterialBoard[i];
            }
            return (short)(material / FV_SCALE);
        }
    }
}
