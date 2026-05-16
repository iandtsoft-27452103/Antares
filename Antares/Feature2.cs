using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Antares.Common;

namespace Antares
{
    internal class Feature2
    {
        public static int[] Rev_Sq = { 80, 79, 78, 77, 76, 75, 74, 73, 72,
                                       71, 70, 69, 68, 67, 66, 65, 64, 63,
                                       62, 61, 60, 59, 58, 57, 56, 55, 54,
                                       53, 52, 51, 50, 49, 48, 47, 46, 45,
                                       44, 43, 42, 41, 40, 39, 38, 37, 36,
                                       35, 34, 33, 32, 31, 30, 29, 28, 27,
                                       26, 25, 24, 23, 22, 21, 20, 19, 18,
                                       17, 16, 15, 14, 13, 12, 11, 10, 9,
                                        8,  7,  6,  5,  4,  3,  2,  1, 0 };

        // Apery方式も良いか？

        //KKPとKPPのインデックス
        public const int kkp_hand_pawn = 0;
        public const int kkp_hand_lance = 18;
        public const int kkp_hand_knight = 22;
        public const int kkp_hand_silver = 26;
        public const int kkp_hand_gold = 30;
        public const int kkp_hand_bishop = 34;
        public const int kkp_hand_rook = 36;
        public const int kkp_pawn = 38;
        public const int kkp_lance = 119;
        public const int kkp_knight = 200;
        public const int kkp_silver = 281;
        public const int kkp_gold = 362;
        public const int kkp_bishop = 443;
        public const int kkp_rook = 524;
        public const int kkp_pro_pawn = 605;
        public const int kkp_pro_lance = 686;
        public const int kkp_pro_knight = 767;
        public const int kkp_pro_silver = 848;
        public const int kkp_horse = 929;
        public const int kkp_dragon = 1010;
        public const int kkp_end = 1091;

        public const int b_hand_pawn = 0;
        public const int b_hand_lance = 18;
        public const int b_hand_knight = 22;
        public const int b_hand_silver = 26;
        public const int b_hand_gold = 30;
        public const int b_hand_bishop = 34;
        public const int b_hand_rook = 36;
        public const int b_pawn = 38;
        public const int b_lance = 119;
        public const int b_knight = 200;
        public const int b_silver = 281;
        public const int b_gold = 362;
        public const int b_bishop = 443;
        public const int b_rook = 524;
        public const int b_pro_pawn = 605;
        public const int b_pro_lance = 686;
        public const int b_pro_knight = 767;
        public const int b_pro_silver = 848;
        public const int b_horse = 929;
        public const int b_dragon = 1010;
        public const int w_hand_pawn = 1091;
        public const int w_hand_lance = 1109;
        public const int w_hand_knight = 1113;
        public const int w_hand_silver = 1117;
        public const int w_hand_gold = 1121;
        public const int w_hand_bishop = 1125;
        public const int w_hand_rook = 1127;
        public const int w_pawn = 1129;
        public const int w_lance = 1210;
        public const int w_knight = 1291;
        public const int w_silver = 1372;
        public const int w_gold = 1453;
        public const int w_bishop = 1534;
        public const int w_rook = 1615;
        public const int w_pro_pawn = 1696;
        public const int w_pro_lance = 1777;
        public const int w_pro_knight = 1858;
        public const int w_pro_silver = 1939;
        public const int w_horse = 2020;
        public const int w_dragon = 2101;
        public const int pp_end = 2182;

        public static int[] kkp_start_index = { -1, kkp_pawn, kkp_lance, kkp_knight, kkp_silver, kkp_gold, kkp_bishop, kkp_rook, -1, kkp_pro_pawn, kkp_pro_lance, kkp_pro_knight, kkp_pro_silver, -1, kkp_horse, kkp_dragon };
        public static int[] kkp_hand_start_index = { -1, kkp_hand_pawn, kkp_hand_lance, kkp_hand_knight, kkp_hand_silver, kkp_hand_gold, kkp_hand_bishop, kkp_hand_rook };
        public static int[,] pp_start_index = { {0, b_pawn, b_lance, b_knight, b_silver, b_gold, b_bishop, b_rook, 0, b_pro_pawn, b_pro_lance, b_pro_knight, b_pro_silver, 0, b_horse, b_dragon},
                                         {0, w_pawn, w_lance, w_knight, w_silver, w_gold, w_bishop, w_rook, 0, w_pro_pawn, w_pro_lance, w_pro_knight, w_pro_silver, 0, w_horse, w_dragon } };
        public static int[,] pp_hand_start_index = { { 0, b_hand_pawn, b_hand_lance, b_hand_knight, b_hand_silver, b_hand_gold, b_hand_bishop, b_hand_rook }, { 0, w_hand_pawn, w_hand_lance, w_hand_knight, w_hand_silver, w_hand_gold, w_hand_bishop, w_hand_rook } };
        public static int[,,] kkp_index_table = new int[Color_NB, Piece_NB, Square_NB];
        public static int[,,] pp_index_table = new int[Color_NB, Piece_NB, Square_NB];

        public static short[,,] fv_kkp = new short[Square_NB, Square_NB, kkp_end];// 非Apery方式で仮置き。
        public static short[,,] fv_kpp = new short[Square_NB, pp_end, pp_end];

        public const string file_name_kkp = "fv_kkp.bin";
        public const string file_name_kpp = "fv_kpp.bin";

        public static void InitKKPIndex()
        {
            for (int c = 0; c < Color_NB; c++)
            {
                for (int pc = (int)Piece.Pawn; pc < Piece_NB; pc++)
                {
                    if (pc == (int)Piece.Gold + Promote || pc == (int)Piece.King)
                        continue;

                    for (int sq = 0; sq < Square_NB; sq++)
                    {
                        int pos = sq;
                        if (c == (int)Common.Color.White)
                            pos = Rev_Sq[pos];
                        kkp_index_table[c, pc, sq] = kkp_start_index[pc] + pos;
                        pp_index_table[c, pc, sq] = pp_start_index[c, pc] + pos;
                    }
                }
            }
        }

        // ランダムに初期化する。差分計算のテスト用。
        public static void RandomInit()
        {
            for (int i = 0; i < Square_NB; i++)
            {
                for (int j = 0; j < Square_NB; j++)
                {
                    for (int k = 0; k < kkp_end; k++)
                    {
                        Random rdm = new Random();
                        int v = rdm.Next(0, 32);
                        fv_kkp[i, j, k] = (short)v;
                    }
                }
            }

            for (int i = 0; i < Square_NB; i++)
            {
                for (int j = 0; j < pp_end; j++)
                {
                    for (int k = 0; k < pp_end; k++)
                    {
                        Random rdm = new Random();
                        int v = rdm.Next(0, 32);
                        fv_kpp[i, j, k] = (short)v;
                    }
                }
            }
        }

        public static void Init()
        {
            fv_kkp = new short[Square_NB, Square_NB, kkp_end];
            fv_kpp = new short[Square_NB, pp_end, pp_end];
        }

        public static void Save()
        {
            using (BinaryWriter writer = new BinaryWriter(System.IO.File.OpenWrite(file_name_kkp)))
            {
                for (int i = 0; i < Square_NB; i++)
                {
                    for (int j = 0; j < Square_NB; j++)
                    {
                        for (int k = 0; k < kkp_end; k++)
                        {
                            writer.Write(fv_kkp[i, j, k]);
                        }
                    }
                }
            }
            using (BinaryWriter writer = new BinaryWriter(System.IO.File.OpenWrite(file_name_kpp)))
            {
                for (int i = 0; i < Square_NB; i++)
                {
                    for (int j = 0; j < pp_end; j++)
                    {
                        for (int k = 0; k < pp_end; k++)
                        {
                            writer.Write(fv_kpp[i, j, k]);
                        }
                    }
                }
            }
        }

        public static void Load()
        {
            //int max_value = 0;
            //int counter = 0;
            using (BinaryReader reader = new BinaryReader(System.IO.File.OpenRead(file_name_kkp)))
            {
                for (int i = 0; i < Square_NB; i++)
                {
                    for (int j = 0; j < Square_NB; j++)
                    {
                        for (int k = 0; k < kkp_end; k++)
                        {
                            fv_kkp[i, j, k] = reader.ReadInt16();
                            //counter++;
                            /*if (fv_kkp[i, j, k] > max_value)
                            {
                                max_value = fv_kkp[i, j, k];
                            }*/
                        }
                    }
                }
            }

            //Console.WriteLine("max_value_kkp = " + max_value.ToString());

            //max_value = 0;
            //counter = 0;
            using (BinaryReader reader = new BinaryReader(System.IO.File.OpenRead(file_name_kpp)))
            {
                for (int i = 0; i < Square_NB; i++)
                {
                    for (int j = 0; j < pp_end; j++)
                    {
                        for (int k = 0; k < pp_end; k++)
                        {
                            fv_kpp[i, j, k] = reader.ReadInt16();
                            /*if (fv_kpp[i, j, k] > max_value)
                            {
                                max_value = fv_kpp[i, j, k];
                            }
                            counter++;*/
                            //if (counter < 200)
                                //Console.WriteLine(fv_kpp[i, j, k]);
                        }
                    }
                }
            }
            //Console.WriteLine("max_value_kpp = " + max_value.ToString());
        }
    }
}
