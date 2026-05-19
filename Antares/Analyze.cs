using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Threading.Tasks;
using static Antares.AlphaBeta7;
using static Antares.Analyze;
using static Antares.AttacksOperation;
using static Antares.BitOperation;
using static Antares.Board;
using static Antares.Common;
using static Antares.CSA;
using static Antares.Evaluate;
using static Antares.Feature2;
using static Antares.GenMoves;
using static Antares.Hash;
using static Antares.Mate1Ply;
using static Antares.Playout;
using static Antares.SFEN;
using static Antares.Test;
using static ICSharpCode.SharpZipLib.Zip.ExtendedUnixData;
using static System.Net.Mime.MediaTypeNames;
using static Tensorboard.TensorShapeProto.Types;
using BitBoard = System.UInt128;

namespace Antares
{
    public static class Analyze
    {
        public struct Tree
        {
            public int win_count;
            public int lose_count;
            public int draw_count;
            public List<Move> temp_moves;
            public List<int> list_win;
            public List<int> list_lose;
            public List<int> list_draw;
            public Tree()
            {
                win_count = 0;
                lose_count = 0;
                draw_count = 0;
                temp_moves = new List<Move>();
                list_win = new List<int>();
                list_lose = new List<int>();
                list_draw = new List<int>();
            }
        }

        public struct Sessions
        {
            public InferenceSession policy_session;
            public InferenceSession value_session;
        }

        public static Sessions sessions;

        public static Piece[] PieceList = { Piece.Pawn, Piece.Lance, Piece.Knight, Piece.Silver, Piece.Gold, Piece.Bishop, Piece.Rook, Piece.King, Piece.Pro_Pawn, Piece.Pro_Lance, Piece.Pro_Knight, Piece.Pro_Silver, Piece.Horse, Piece.Dragon };
        public static int[] HandLimit = { 0, 18, 4, 4, 4, 4, 2, 2 };

        static void Init()
        {
            IniRand(5489U);
            IniRandomTable();
            InitObs();
            InitPieceAttacks();
            InitLongAttacks();
            InitPieceTable();
        }

        public static void LoadModel()
        {
            // Path to your ONNX model
            string modelPath = "model.onnx";
            string modelPathV = "model_value.onnx";

            // Validate model file
            if (!System.IO.File.Exists(modelPath))
            {
                Console.WriteLine($"Error: Model file not found at {modelPath}");
                return;
            }
            try
            {
                // Create inference session
                sessions.policy_session = new InferenceSession(modelPath);
                sessions.value_session = new InferenceSession(modelPathV);

                // Example input: 1D tensor with 4 float values
                // Replace with your actual input shape and data
                //var inputData = new float[] { 5.1f, 3.5f, 1.4f, 0.2f };
                var inputData = new float[105 * 81];
                var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 105, 9, 9 });

                // Get model input name
                string inputName = sessions.policy_session.InputMetadata.Keys.First();
                string inputNameV = sessions.value_session.InputMetadata.Keys.First();

                // Create input container
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
                };

                // Run inference
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = sessions.policy_session.Run(inputs);
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> resultsV = sessions.value_session.Run(inputs);

                // Process output
                /*foreach (var result in results)
                {
                    Console.WriteLine($"Output: {result.Name}");
                    if (result.Value is IEnumerable<float> outputValues)
                    {
                        Console.WriteLine(string.Join(", ", outputValues));
                    }
                    else
                    {
                        Console.WriteLine("Output type not recognized.");
                    }
                }*/
                // Process output
                /*foreach (var result in resultsV)
                {
                    Console.WriteLine($"Output: {result.Name}");
                    if (result.Value is IEnumerable<float> outputValues)
                    {
                        Console.WriteLine(string.Join(", ", outputValues));
                    }
                    else
                    {
                        Console.WriteLine("Output type not recognized.");
                    }
                }*/
            }
            catch (OnnxRuntimeException ex)
            {
                Console.WriteLine($"ONNX Runtime error: {ex.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"General error: {ex.Message}");
            }
        }

        public static float[] MakeInputFeature(ref BoardTree BTree, int color)
        {
            float[] inputs = new float[1*105*81];
            int index = 0;
            // 駒の配置 先後で14枚ずつ
            for (int c = 0; c < Color_NB; c++)
            {
                for (int i = 0; i < PieceList.Length; i++)
                {
                    int pc = (int)PieceList[i];
                    BitBoard bb_piece = BTree.BB_Piece[c, pc];
                    while (bb_piece > 0)
                    {
                        int sq = Square(bb_piece);
                        bb_piece ^= ABB_Mask[sq];
                        inputs[index + sq] = 1.0f;
                    }
                    index += Square_NB;
                }

                for (int pc = (int)Piece.Pawn; pc <= Piece_Can_Drop_NB; pc++)
                {
                    int n = (BTree.Hand[c] & Hand_Mask[pc]) >> Hand_Rev_Bit[pc];

                    for (int i = 0; i < n; i++)
                    {
                        Array.Fill(inputs, 1.0F, index, Square_NB);
                        index += Square_NB;
                    }
                    for (int i = n; i < HandLimit[pc]; i++)
                    {
                        index += Square_NB;
                    }
                }
            }

            // 手番 1枚
            if (color == 0)
            {
                Array.Fill(inputs, 1.0F, index, Square_NB);
            }
            return inputs;
        }

        public static float[,] MakeOutputLabel(Move move, ref Direction d, ref int idirec)
        {
            const int output_label_num = 32;
            const int delta = 12;
            float[,] outputs = new float[output_label_num, Square_NB];
            if (move.From >= Square_NB)
            {
                switch (move.PieceType)
                {
                    case Piece.Pawn:
                        outputs[25, move.To] = 1.0f;
                        idirec = 25;
                        break;
                    case Piece.Lance:
                        outputs[26, move.To] = 1.0f;
                        idirec = 26;
                        break;
                    case Piece.Knight:
                        outputs[27, move.To] = 1.0f;
                        idirec = 27;
                        break;
                    case Piece.Silver:
                        outputs[28, move.To] = 1.0f;
                        idirec = 28;
                        break;
                    case Piece.Gold:
                        outputs[29, move.To] = 1.0f;
                        idirec = 29;
                        break;
                    case Piece.Bishop:
                        outputs[30, move.To] = 1.0f;
                        idirec = 30;
                        break;
                    case Piece.Rook:
                        outputs[31, move.To] = 1.0f;
                        idirec = 31;
                        break;
                }
            }
            else
            {
                Direction direc = Adirec[move.From, move.To];
                d = direc;
                if (move.FlagPromo == 0)
                {
                    switch (direc)
                    {
                        case Direction.Direc_Diag1_U2d:
                            outputs[6, move.To] = 1.0f;
                            idirec = 6;
                            break;
                        case Direction.Direc_Diag1_D2u:
                            outputs[3, move.To] = 1.0f;
                            idirec = 3;
                            break;
                        case Direction.Direc_Diag2_U2d:
                            outputs[8, move.To] = 1.0f;
                            idirec = 8;
                            break;
                        case Direction.Direc_Diag2_D2u:
                            outputs[1, move.To] = 1.0f;
                            idirec = 1;
                            break;
                        case Direction.Direc_File_U2d:
                            outputs[7, move.To] = 1.0f;
                            idirec = 7;
                            break;
                        case Direction.Direc_File_D2u:
                            outputs[2, move.To] = 1.0f;
                            idirec = 2;
                            break;
                        case Direction.Direc_Rank_L2r:
                            outputs[5, move.To] = 1.0f;
                            idirec = 5;
                            break;
                        case Direction.Direc_Rank_R2l:
                            outputs[4, move.To] = 1.0f;
                            idirec = 4;
                            break;
                        case Direction.Direc_Knight_L_U2d:
                            outputs[12, move.To] = 1.0f;
                            idirec = 12;
                            break;
                        case Direction.Direc_Knight_R_U2d:
                            outputs[11, move.To] = 1.0f;
                            idirec = 11;
                            break;
                        case Direction.Direc_Knight_L_D2u:
                            outputs[10, move.To] = 1.0f;
                            idirec = 10;
                            break;
                        case Direction.Direc_Knight_R_D2u:
                            outputs[9, move.To] = 1.0f;
                            idirec = 9;
                            break;
                    }
                }
                else
                {
                    switch (direc)
                    {
                        case Direction.Direc_Diag1_U2d:
                            outputs[6 + delta, move.To] = 1.0f;
                            idirec = 6 + delta;
                            break;
                        case Direction.Direc_Diag1_D2u:
                            outputs[3 + delta, move.To] = 1.0f;
                            idirec = 3 + delta;
                            break;
                        case Direction.Direc_Diag2_U2d:
                            outputs[8 + delta, move.To] = 1.0f;
                            idirec = 8 + delta;
                            break;
                        case Direction.Direc_Diag2_D2u:
                            outputs[1 + delta, move.To] = 1.0f;
                            idirec = 1 + delta;
                            break;
                        case Direction.Direc_File_U2d:
                            outputs[7 + delta, move.To] = 1.0f;
                            idirec = 7 + delta;
                            break;
                        case Direction.Direc_File_D2u:
                            outputs[2 + delta, move.To] = 1.0f;
                            idirec = 2 + delta;
                            break;
                        case Direction.Direc_Rank_L2r:
                            outputs[5 + delta, move.To] = 1.0f;
                            idirec = 5 + delta;
                            break;
                        case Direction.Direc_Rank_R2l:
                            outputs[4 + delta, move.To] = 1.0f;
                            idirec = 4 + delta;
                            break;
                        case Direction.Direc_Knight_L_U2d:
                            outputs[12 + delta, move.To] = 1.0f;
                            idirec = 12 + delta;
                            break;
                        case Direction.Direc_Knight_R_U2d:
                            outputs[11 + delta, move.To] = 1.0f;
                            idirec = 11 + delta;
                            break;
                        case Direction.Direc_Knight_L_D2u:
                            outputs[10 + delta, move.To] = 1.0f;
                            idirec = 10 + delta;
                            break;
                        case Direction.Direc_Knight_R_D2u:
                            outputs[9 + delta, move.To] = 1.0f;
                            idirec = 9 + delta;
                            break;
                    }
                }
            }
            return outputs;
        }

        public static string[] ExecPolicy(string[] str_sfen)
        {
            int batch_size = str_sfen.Length;
            int[] newShape = new int[] { batch_size, 105, 9, 9 };
            int[] newShape2 = new int[] { batch_size, 32, 9, 9 };
            string[] str_out = new string[batch_size];
            float[] inputData = new float[1 * 105 * 9 * 9];
            float[] inputData2 = new float[batch_size * 105 * 9 * 9];
            Tensor<float> outputTensor = new DenseTensor<float>(new int[] { batch_size, 32, 81 });
            BoardTree bt = new BoardTree();
            for (int i = 0; i < batch_size; i++)
            {
                bt = ToBoard(str_sfen[i]);
                inputData = MakeInputFeature(ref bt, (int)bt.RootColor);
                inputData.CopyTo(inputData2, i * 1 * 105 * 9 * 9);
                //inputData2[i] = inputData[0];
            }
            DenseTensor<float> inputTensor = new DenseTensor<float>(inputData2, newShape);
            string inputName = sessions.policy_session.InputMetadata.Keys.First();
            // Create input container
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = sessions.policy_session.Run(inputs);
            outputTensor = results.First().AsTensor<float>();
            outputTensor = outputTensor.Reshape(newShape2);
            for (int i = 0; i < batch_size; i++)
            {
                bt = ToBoard(str_sfen[i]);
                List<Move> moves = new List<Move>();
                if (IsAttacked(bt, bt.SQ_King[(int)bt.RootColor], (int)bt.RootColor) == 0)
                {
                    GenCap(bt, (int)bt.RootColor, ref moves);
                    GenNoCap(bt, (int)bt.RootColor, ref moves);
                    GenDrop(bt, (int)bt.RootColor, ref moves);
                }
                else
                {
                    GenEvasion(bt, (int)bt.RootColor, ref moves);
                }
                List<Move> legal_moves = new List<Move>();
                for (int j = 0; j < moves.Count; j++)
                {
                    int ifrom = moves[j].From;
                    int ito = moves[j].To;
                    if (ifrom < Square_NB)
                    {
                        // Discovered Checkの場合
                        if (IsPinnedOnKing(bt, ifrom, Adirec[ifrom, ito], (int)bt.RootColor) > 0)
                        {
                            continue;
                        }
                    }
                    // 玉を取ってしまう手
                    if (moves[j].CapPiece == Piece.King)
                    {
                        continue;
                    }
                    legal_moves.Add(moves[j]);
                }
                //float v_sum = 0.0f;
                List<float> li_v = new List<float>();
                List<string> str_moves = new List<string>();
                for (int j = 0; j < legal_moves.Count; j++)
                {
                    string s = Move2CSA(legal_moves[j]);
                    int ifrom = moves[j].From;
                    int ito = moves[j].To;
                    Direction d = new Direction();
                    //int idirec = (int)Adirec[ifrom, ito];
                    int idirec = 0;
                    var lbl = MakeOutputLabel(legal_moves[j], ref d, ref idirec);
                    float v = 0.0f;
                    try
                    {
                        v = outputTensor[i, idirec, legal_moves[j].To];
                    }
                    catch (Exception e)
                    {
                        v = 0.0f;
                    }
                   
                    //v_sum += v;
                    str_moves.Add(s);
                    li_v.Add(v);
                }

                List<int> idxes = new List<int>();
                int limit = 4;
                if (legal_moves.Count <= limit)
                {
                    limit = legal_moves.Count;
                }

                List<float> outputs = new List<float>();
                for (int j = 0; j < limit; j++)
                {
                    int max_index = li_v.IndexOf(li_v.Max());
                    idxes.Add(max_index);
                    float temp_v = li_v[idxes[j]];
                    outputs.Add(temp_v);
                    li_v[max_index] = float.MinValue;
                }
                List<Move> moves2 = new List<Move>();
                for (int j = 0; j < idxes.Count; j++)
                {
                    moves2.Add(legal_moves[idxes[j]]);
                }

                //List<float> outputs = new List<float>();
                for (int j = 0; j < moves2.Count; j++)
                {
                    if (j > 0)
                    {
                        str_out[i] += ",";
                    }
                    float v = li_v[j] / li_v.Sum();
                    outputs.Add(v);
                    str_out[i] += str_moves[j] + " " + v.ToString();
                }
            }
            return str_out;
        }

        public static string[] ExecValue(string[] str_sfen)
        {
            int batch_size = str_sfen.Length;
            int[] newShape = new int[] { batch_size, 105, 9, 9 };
            string[] str_out = new string[batch_size];
            float[] inputData = new float[1 * 105 * 9 * 9];
            float[] inputData2 = new float[batch_size * 105 * 9 * 9];
            BoardTree bt = new BoardTree();
            for (int i = 0; i <  batch_size; i++)
            {
                bt = ToBoard(str_sfen[i]);
                inputData = MakeInputFeature(ref bt, (int)bt.RootColor);
                inputData.CopyTo(inputData2, i * 1 * 105 * 9 * 9);
                //inputData2[i] = inputData[0];
            }
            DenseTensor<float> inputTensor = new DenseTensor<float>(inputData2, newShape);
            string inputNameV = sessions.value_session.InputMetadata.Keys.First();
            // Create input container
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputNameV, inputTensor)
            };
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> resultsV = sessions.value_session.Run(inputs);
            float[] outputTensor = resultsV.First().AsEnumerable<float>().ToArray();
            for (int i = 0; i < batch_size; i++)
            {
                str_out[i] = Sigmoid(outputTensor[i]).ToString();
            }

            return str_out;
        }

        public static float Sigmoid(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }

        // ロールアウトポリシーなしバージョン。
        public static void AnalyzeWithDeepCNN(int num_tasks, int num_mate_tasks, int thinking_time, int mate_search_depth, string analyze_file_name, string record_file_name,
                             string str_game_date, string str_match_name, string str_black_player, string str_white_player)
        {
            //const int num_threads = 6;

            StreamWriter sw = IO.OpenStreamWriter(analyze_file_name);
            sw.WriteLine("対局日：" + str_game_date);
            sw.WriteLine();
            sw.WriteLine("棋戦名：" + str_match_name);
            sw.WriteLine();
            sw.WriteLine("先手：" + str_black_player);
            sw.WriteLine();
            sw.WriteLine("後手：" + str_white_player);
            sw.WriteLine();

            LoadModel();
            var temp = IO.ReadRecordFile(record_file_name);
            Record r = temp[0];
            BoardTree bt = new BoardTree();
            Board.BoardTreeAlloc(ref bt);
            Board.Init(ref bt);
            int color = 0;
            string str_result = "";
            string str_mate_pv = "";
            string str_first_move = "";
            int denomi = r.str_moves.Count();
            int denomi_black = 0;
            int denomi_white = 0;
            int[] move_first_accuracy = new int[2];
            int[] move_second_accuracy = new int[2];
            int[] move_third_accuracy = new int[2];
            Move mate_first_move = new Move();
            for (int i = 0; i < r.str_moves.Count(); i++)
            {
                /*if (i != r.str_moves.Count() - 1)
                {
                    Do(ref bt, CSA2Move(bt, r.str_moves[i]), color);
                    color ^= 1;
                    continue;
                }*/
                bt.RootColor = (Antares.Common.Color)color;
                str_result = "";
                str_mate_pv = "";
                str_first_move = "";
                mate_first_move.Clear();
                Console.WriteLine("ply = " + (i + 1).ToString());
                SearchWrapper(ref bt, num_tasks, num_mate_tasks, thinking_time, mate_search_depth, ref str_result, ref str_mate_pv, r.str_moves[i], ref move_first_accuracy, ref move_second_accuracy, ref move_third_accuracy, ref mate_first_move, ref str_first_move);
                if (str_mate_pv != "")
                {
                    string str_accurate = "×";
                    string str_color = "+";
                    if (color == (int)Antares.Common.Color.White)
                    {
                        str_color = "-";
                    }
                    if (r.str_moves[i] == Move2CSA(mate_first_move))
                    {
                        str_accurate = "○";
                        move_first_accuracy[color] += 1;
                    }
                    else
                    {
                        string s = str_mate_pv.Substring(1, 6);
                        if (r.str_moves[i] == s)
                        {
                            str_accurate = "○";
                            move_first_accuracy[color] += 1;
                        }
                    }
                    str_mate_pv = "ply=" + (i + 1).ToString() + ", 棋譜の手: " + str_color + r.str_moves[i] + ", result= " + str_accurate + ", 詰みあり： " + str_mate_pv;
                    sw.WriteLine(str_mate_pv);
                }
                else
                {
                    string str_color = "+";
                    if (color == (int)Antares.Common.Color.White)
                    {
                        str_color = "-";
                    }
                    /*if (r.str_moves[i] == Move2CSA(mate_first_move))
                    {
                        move_first_accuracy[color] += 1;
                    }
                    else
                    {
                        string s = str_first_move;
                        if (r.str_moves[i] == s)
                        {
                            move_first_accuracy[color] += 1;
                        }
                    }*/
                    str_result = "ply=" + (i + 1).ToString() + ", 棋譜の手: " + str_color + r.str_moves[i] + ", " + str_result;
                    sw.WriteLine(str_result);
                }
                Move m = CSA2Move(bt, r.str_moves[i]);
                Do(ref bt, m, color);
                color ^= 1;
                if (color == (int)Antares.Common.Color.Black)
                {
                    denomi_black += 1;
                }
                else
                {
                    denomi_white += 1;
                }
            }
            sw.WriteLine();

            int temp_n;
            string temp_s;
            temp_n = move_first_accuracy[0];
            temp_s = temp_n.ToString() + " / " + denomi_black.ToString() + " = " + ((float)temp_n / (float)denomi_black).ToString("P2");
            sw.WriteLine("先手一致率：" + temp_s);
            sw.WriteLine();
            temp_n = move_first_accuracy[1];
            temp_s = temp_n.ToString() + " / " + denomi_white.ToString() + " = " + ((float)temp_n / (float)denomi_white).ToString("P2");
            sw.WriteLine("後手一致率：" + temp_s);
            sw.WriteLine();
            temp_n = move_first_accuracy[0] + move_first_accuracy[1];
            temp_s = temp_n.ToString() + " / " + denomi.ToString() + " = " + ((float)temp_n / (float)denomi).ToString("P2");
            sw.WriteLine("全体一致率：" + temp_s);
            sw.WriteLine();
            temp_n = move_first_accuracy[0] + move_second_accuracy[0] + move_third_accuracy[0] + move_first_accuracy[0] + move_second_accuracy[0] + move_third_accuracy[0];
            temp_s = temp_n.ToString() + " / " + denomi.ToString() + " = " + ((float)temp_n / (float)denomi).ToString("P2");
            sw.WriteLine("3位以内の確率：" + temp_s);
            sw.WriteLine();
            sw.WriteLine("解析エンジン名：Antares Ver.1.0.0");
            sw.Close();
        }

        // ロールアウトポリシーありバージョン。
        public static void AnalyzeWithDeepCNN2(int num_tasks, int num_mate_tasks, int thinking_time, int mate_search_depth, string analyze_file_name, string record_file_name,
                     string str_game_date, string str_match_name, string str_black_player, string str_white_player)
        {
            //const int num_threads = 6;

            StreamWriter sw = IO.OpenStreamWriter("analyze_result_using_deep_cnn_with_rollout_policy.txt");
            sw.WriteLine("対局日：" + str_game_date);
            sw.WriteLine();
            sw.WriteLine("棋戦名：" + str_match_name);
            sw.WriteLine();
            sw.WriteLine("先手：" + str_black_player);
            sw.WriteLine();
            sw.WriteLine("後手：" + str_white_player);
            sw.WriteLine();

            LoadModel();
            var temp = IO.ReadRecordFile(record_file_name);
            Record r = temp[0];
            BoardTree bt = new BoardTree();
            Board.BoardTreeAlloc(ref bt);
            Board.Init(ref bt);
            int color = 0;
            string str_result = "";
            string str_mate_pv = "";
            string str_first_move = "";
            int denomi = r.str_moves.Count();
            int denomi_black = 0;
            int denomi_white = 0;
            int[] move_first_accuracy = new int[2];
            int[] move_second_accuracy = new int[2];
            int[] move_third_accuracy = new int[2];
            Move mate_first_move = new Move();
            for (int i = 0; i < r.str_moves.Count(); i++)
            {
                /*if (i != r.str_moves.Count() - 1)
                {
                    Do(ref bt, CSA2Move(bt, r.str_moves[i]), color);
                    color ^= 1;
                    continue;
                }*/
                bt.RootColor = (Antares.Common.Color)color;
                str_result = "";
                str_mate_pv = "";
                str_first_move = "";
                mate_first_move.Clear();
                Console.WriteLine("ply = " + (i + 1).ToString());
                SearchWrapper2(ref bt, num_tasks, num_mate_tasks, thinking_time, mate_search_depth, ref str_result, ref str_mate_pv, r.str_moves[i], ref move_first_accuracy, ref move_second_accuracy, ref move_third_accuracy, ref mate_first_move, ref str_first_move);
                if (str_mate_pv != "")
                {
                    string str_accurate = "×";
                    string str_color = "+";
                    if (color == (int)Antares.Common.Color.White)
                    {
                        str_color = "-";
                    }
                    if (r.str_moves[i] == Move2CSA(mate_first_move))
                    {
                        str_accurate = "○";
                        move_first_accuracy[color] += 1;
                    }
                    else
                    {
                        string s = str_mate_pv.Substring(1, 6);
                        if (r.str_moves[i] == s)
                        {
                            str_accurate = "○";
                            move_first_accuracy[color] += 1;
                        }
                    }
                    str_mate_pv = "ply=" + (i + 1).ToString() + ", 棋譜の手: " + str_color + r.str_moves[i] + ", result= " + str_accurate + ", 詰みあり： " + str_mate_pv;
                    sw.WriteLine(str_mate_pv);
                }
                else
                {
                    string str_color = "+";
                    if (color == (int)Antares.Common.Color.White)
                    {
                        str_color = "-";
                    }
                    /*if (r.str_moves[i] == Move2CSA(mate_first_move))
                    {
                        move_first_accuracy[color] += 1;
                    }
                    else
                    {
                        string s = str_first_move;
                        if (r.str_moves[i] == s)
                        {
                            move_first_accuracy[color] += 1;
                        }
                    }*/
                    str_result = "ply=" + (i + 1).ToString() + ", 棋譜の手: " + str_color + r.str_moves[i] + ", " + str_result;
                    sw.WriteLine(str_result);
                }
                Move m = CSA2Move(bt, r.str_moves[i]);
                Do(ref bt, m, color);
                color ^= 1;
                if (color == (int)Antares.Common.Color.Black)
                {
                    denomi_black += 1;
                }
                else
                {
                    denomi_white += 1;
                }
            }
            sw.WriteLine();

            int temp_n;
            string temp_s;
            temp_n = move_first_accuracy[0];
            temp_s = temp_n.ToString() + " / " + denomi_black.ToString() + " = " + ((float)temp_n / (float)denomi_black).ToString("P2");
            sw.WriteLine("先手一致率：" + temp_s);
            sw.WriteLine();
            temp_n = move_first_accuracy[1];
            temp_s = temp_n.ToString() + " / " + denomi_white.ToString() + " = " + ((float)temp_n / (float)denomi_white).ToString("P2");
            sw.WriteLine("後手一致率：" + temp_s);
            sw.WriteLine();
            temp_n = move_first_accuracy[0] + move_first_accuracy[1];
            temp_s = temp_n.ToString() + " / " + denomi.ToString() + " = " + ((float)temp_n / (float)denomi).ToString("P2");
            sw.WriteLine("全体一致率：" + temp_s);
            sw.WriteLine();
            temp_n = move_first_accuracy[0] + move_second_accuracy[0] + move_third_accuracy[0] + move_first_accuracy[0] + move_second_accuracy[0] + move_third_accuracy[0];
            temp_s = temp_n.ToString() + " / " + denomi.ToString() + " = " + ((float)temp_n / (float)denomi).ToString("P2");
            sw.WriteLine("3位以内の確率：" + temp_s);
            sw.WriteLine();
            sw.WriteLine("解析エンジン名：Antares Ver.1.0.0");
            sw.Close();
        }


        public static void SearchWrapper(ref BoardTree bt, int num_tasks, int num_mate_tasks, int thinking_time, int mate_search_depth, ref string str_result, ref string param_str_mate_pv, string str_record_move,
                              ref int[] move_first_accuracy, ref int[] move_second_accuracy, ref int[] move_third_accurasy, ref Move mate_first_move, ref string str_first_move)
        {
            MCTS4 mcts_tree0 = new MCTS4();
            MCTS4 mcts_tree1 = new MCTS4();
            MCTS4 mcts_tree2 = new MCTS4();
            MCTS4 mcts_tree3 = new MCTS4();
            MCTS4 mcts_tree4 = new MCTS4();
            MCTS4 mcts_tree5 = new MCTS4();
            string str_mate_pv = "";

            for (int i = 0; i < num_tasks; i++)
            {
                switch (i)
                {
                    case 0:
                        mcts_tree0 = new MCTS4();
                        mcts_tree0.TaskNumber = 0;
                        mcts_tree0.BTree = DeepCopy(bt, false);
                        mcts_tree0.GenRootMoves();
                        mcts_tree0.SearchTimeLimit = thinking_time * 1000;
                        mcts_tree0.SetRootOutput();
                        break;
                    case 1:
                        mcts_tree1 = new MCTS4();
                        mcts_tree1.TaskNumber = 1;
                        mcts_tree1.BTree = DeepCopy(bt, false);
                        mcts_tree1.GenRootMoves();
                        mcts_tree1.SearchTimeLimit = thinking_time * 1000;
                        mcts_tree1.SetRootOutput();
                        break;
                    case 2:
                        mcts_tree2 = new MCTS4();
                        mcts_tree2.TaskNumber = 2;
                        mcts_tree2.BTree = DeepCopy(bt, false);
                        mcts_tree2.GenRootMoves();
                        mcts_tree2.SearchTimeLimit = thinking_time * 1000;
                        mcts_tree2.SetRootOutput();
                        break;
                    case 3:
                        mcts_tree3 = new MCTS4();
                        mcts_tree3.TaskNumber = 3;
                        mcts_tree3.BTree = DeepCopy(bt, false);
                        mcts_tree3.GenRootMoves();
                        mcts_tree3.SearchTimeLimit = thinking_time * 1000;
                        mcts_tree3.SetRootOutput();
                        break;
                    case 4:
                        mcts_tree4 = new MCTS4();
                        mcts_tree4.TaskNumber = 4;
                        mcts_tree4.BTree = DeepCopy(bt, false);
                        mcts_tree4.GenRootMoves();
                        mcts_tree4.SearchTimeLimit = thinking_time * 1000;
                        mcts_tree4.SetRootOutput();
                        break;
                    case 5:
                        mcts_tree5 = new MCTS4();
                        mcts_tree5.TaskNumber = 5;
                        mcts_tree5.BTree = DeepCopy(bt, false);
                        mcts_tree5.GenRootMoves();
                        mcts_tree5.SearchTimeLimit = thinking_time * 1000;
                        mcts_tree5.SetRootOutput();
                        break;
                }
            }
            List<Move> checkMoves = new List<Move>();
            List<Move> li_checkMoves0 = new List<Move>();
            List<Move> li_checkMoves1 = new List<Move>();
            Mate mst0 = new Mate();
            Mate mst1 = new Mate();

            GenCheck(bt, (int)bt.RootColor, ref checkMoves);
            if (checkMoves.Count > 0)
            {
                for (int i = 0; i < checkMoves.Count; i++)
                {
                    if (i % 2 == 0)
                    {
                        li_checkMoves0.Add(checkMoves[i]);
                    }
                    else
                    {
                        li_checkMoves1.Add(checkMoves[i]);
                    }
                }
                for (int i = 0; i < num_mate_tasks; i++)
                {
                    switch (i)
                    {
                        case 0:
                            mst0.BTree = DeepCopy(bt, false);
                            mst0.RootCheckMoves = new List<Move>(li_checkMoves0);
                            mst0.max_ply = mate_search_depth;
                            break;
                        case 1:
                            mst1.BTree = DeepCopy(bt, false);
                            mst1.RootCheckMoves = new List<Move>(li_checkMoves1);
                            mst1.max_ply = mate_search_depth;
                            break;
                    }
                }
            }

            Task task0 = new Task(() => mcts_tree0.Root());
            Task task1 = new Task(() => mcts_tree1.Root());
            Task task2 = new Task(() => mcts_tree2.Root());
            Task task3 = new Task(() => mcts_tree3.Root()); ;
            Task task4 = new Task(() => mcts_tree4.Root()); ;
            Task task5 = new Task(() => mcts_tree5.Root());
            for (int i = 0; i < num_tasks; i++)
            {
                switch (i)
                {
                    case 0:
                        //task0 = Task.Run(() => mcts4.mcts_tree0.Root());
                        task0.Start();
                        break;
                    case 1:
                        //task1 = Task.Run(() => mcts4.mcts_tree1.Root());
                        task1.Start();
                        break;
                    case 2:
                        //task2 = Task.Run(() => mcts4.mcts_tree2.Root());
                        task2.Start();
                        break;
                    case 3:
                        //task3 = Task.Run(() => mcts4.mcts_tree3.Root());
                        task3.Start();
                        break;
                    case 4:
                        //task4 = Task.Run(() => mcts4.mcts_tree4.Root());
                        task4.Start();
                        break;
                    case 5:
                        //task5 = Task.Run(() => mcts4.mcts_tree5.Root());
                        task5.Start();
                        break;
                }
            }
            Task task_mate0 = new Task(() => mst0.MateSearchWrapper(mate_search_depth));
            Task task_mate1 = new Task(() => mst1.MateSearchWrapper(mate_search_depth));
            if (checkMoves.Count > 0)
            {
                for (int i = 0; i < num_mate_tasks; i++)
                {
                    switch (i)
                    {
                        case 0:
                            //task_mate0 = Task.Run(() => mst0.MateSearchWrapper());
                            task_mate0.Start();
                            break;
                        case 1:
                            //task_mate1 = Task.Run(() => mst1.MateSearchWrapper());
                            task_mate1.Start();
                            break;
                    }
                }
            }

            int index = 0;
            string[] str_policy_requests = new string[num_tasks];
            string[] str_value_requests = new string[num_tasks];
            for (int i = 0; i < num_tasks; i++)
            {
                str_policy_requests[i] = "";
                str_value_requests[i] = "";
            }
            string temp_s;
            string[] temp_s2;
            bool flag = false;
            //bool is_completed = false;
            int p = 0;
            int v = 0;
            int cnt = 0;
            Stopwatch sw = new Stopwatch();
            sw.Start();
            //int cnt_0 = 0;
            //int cnt_1 = 0;
            long base_time = sw.ElapsedMilliseconds;
            while (true)
            {
                switch (index)
                {
                    case 0:
                        if (mcts_tree0.queue_to_main_thread_p.Count > 0)
                        {
                            temp_s = mcts_tree0.queue_to_main_thread_p.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_policy_requests[index] = temp_s2[1];
                            p += 1;
                        }
                        else if (mcts_tree0.queue_to_main_thread_v.Count > 0)
                        {
                            temp_s = mcts_tree0.queue_to_main_thread_v.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_value_requests[index] = temp_s2[1];
                            v += 1;
                        }
                        break;
                    case 1:
                        if (mcts_tree1.queue_to_main_thread_p.Count > 0)
                        {
                            temp_s = mcts_tree1.queue_to_main_thread_p.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_policy_requests[index] = temp_s2[1];
                            p += 1;
                        }
                        else if (mcts_tree1.queue_to_main_thread_v.Count > 0)
                        {
                            temp_s = mcts_tree1.queue_to_main_thread_v.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_value_requests[index] = temp_s2[1];
                            v += 1;
                        }
                        break;
                    case 2:
                        if (mcts_tree2.queue_to_main_thread_p.Count > 0)
                        {
                            temp_s = mcts_tree2.queue_to_main_thread_p.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_policy_requests[index] = temp_s2[1];
                            p += 1;
                        }
                        else if (mcts_tree2.queue_to_main_thread_v.Count > 0)
                        {
                            temp_s = mcts_tree2.queue_to_main_thread_v.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_value_requests[index] = temp_s2[1];
                            v += 1;
                        }
                        break;
                    case 3:
                        if (mcts_tree3.queue_to_main_thread_p.Count > 0)
                        {
                            temp_s = mcts_tree3.queue_to_main_thread_p.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_policy_requests[index] = temp_s2[1];
                            p += 1;
                        }
                        else if (mcts_tree3.queue_to_main_thread_v.Count > 0)
                        {
                            temp_s = mcts_tree3.queue_to_main_thread_v.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_value_requests[index] = temp_s2[1];
                            v += 1;
                        }
                        break;
                    case 4:
                        if (mcts_tree4.queue_to_main_thread_p.Count > 0)
                        {
                            temp_s = mcts_tree4.queue_to_main_thread_p.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_policy_requests[index] = temp_s2[1];
                            p += 1;
                        }
                        else if (mcts_tree4.queue_to_main_thread_v.Count > 0)
                        {
                            temp_s = mcts_tree4.queue_to_main_thread_v.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_value_requests[index] = temp_s2[1];
                            v += 1;
                        }
                        break;
                    case 5:
                        if (mcts_tree5.queue_to_main_thread_p.Count > 0)
                        {
                            temp_s = mcts_tree5.queue_to_main_thread_p.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_policy_requests[index] = temp_s2[1];
                            p += 1;
                        }
                        else if (mcts_tree5.queue_to_main_thread_v.Count > 0)
                        {
                            temp_s = mcts_tree5.queue_to_main_thread_v.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_value_requests[index] = temp_s2[1];
                            v += 1;
                        }
                        break;
                }
                if (v == num_tasks || p == num_tasks)
                {
                    flag = true;
                }
                //制限時間に達したらis_abortをTrueにする。
                long elapsed = sw.ElapsedMilliseconds;
                if (elapsed - base_time > 300)
                {
                    base_time = elapsed;
                    flag = true;
                }

                if (elapsed > thinking_time * 1000)
                {
                    for (int i = 0; i < num_tasks; i++)
                    {
                        switch (i)
                        {
                            case 0:
                                mcts_tree0.is_abort = true;
                                break;
                            case 1:
                                mcts_tree1.is_abort = true;
                                break;
                            case 2:
                                mcts_tree2.is_abort = true;
                                break;
                            case 3:
                                mcts_tree3.is_abort = true;
                                break;
                            case 4:
                                mcts_tree4.is_abort = true;
                                break;
                            case 5:
                                mcts_tree5.is_abort = true;
                                break;
                        }
                    }
                }

                if (flag == true)
                {
                    while (true)
                    {
                        if (p == 0 && v == 0)
                        {
                            break;
                        }
                        if (p > 0)
                        {
                            string[] str_sfen = new string[p];
                            int idx = 0;
                            for (int i = 0; i < p; i++)
                            {
                                str_sfen[i] = "";
                            }
                            for (int i = 0; i < num_tasks; i++)
                            {
                                if (str_policy_requests[i] != "")
                                {
                                    str_sfen[idx] = str_policy_requests[i];
                                    idx += 1;
                                }
                                if (idx == p)
                                {
                                    break;
                                }
                            }
                            string[] str_ret = ExecPolicy(str_sfen);
                            idx = 0;
                            for (int i = 0; i < num_tasks; i++)
                            {
                                if (str_policy_requests[i] != "")
                                {


                                    switch (i)
                                    {
                                        case 0:
                                            mcts_tree0.queue_from_main_thread_p.Enqueue(str_ret[idx]);
                                            break;
                                        case 1:
                                            mcts_tree1.queue_from_main_thread_p.Enqueue(str_ret[idx]);
                                            break;
                                        case 2:
                                            mcts_tree2.queue_from_main_thread_p.Enqueue(str_ret[idx]);
                                            break;
                                        case 3:
                                            mcts_tree3.queue_from_main_thread_p.Enqueue(str_ret[idx]);
                                            break;
                                        case 4:
                                            mcts_tree4.queue_from_main_thread_p.Enqueue(str_ret[idx]);
                                            break;
                                        case 5:
                                            mcts_tree5.queue_from_main_thread_p.Enqueue(str_ret[idx]);
                                            break;
                                    }
                                    idx += 1;
                                }
                                if (idx == p)
                                {
                                    break;
                                }
                            }
                            p = 0;
                            break;
                        }
                        if (v > 0)
                        {
                            string[] str_sfen = new string[v];
                            int idx = 0;
                            for (int i = 0; i < v; i++)
                            {
                                str_sfen[i] = "";
                            }
                            for (int i = 0; i < num_tasks; i++)
                            {
                                if (str_value_requests[i] != "")
                                {
                                    str_sfen[idx] = str_value_requests[i];
                                    idx += 1;
                                }
                                if (idx == v)
                                {
                                    break;
                                }
                            }
                            string[] str_ret;
                            try
                            {
                                str_ret = ExecValue(str_sfen);
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine("例外が発生しましたが、処理を続行します。");
                                int temp_l = str_sfen.Length;
                                str_ret = new string[temp_l];
                                for (int j = 0; j < temp_l; j++)
                                {
                                    str_ret[j] = "0.0";
                                }
                            }
                            idx = 0;
                            for (int i = 0; i < num_tasks; i++)
                            {
                                if (str_value_requests[i] != "")
                                {
                                    switch (i)
                                    {
                                        case 0:
                                            mcts_tree0.queue_from_main_thread_v.Enqueue(str_ret[idx]);
                                            break;
                                        case 1:
                                            mcts_tree1.queue_from_main_thread_v.Enqueue(str_ret[idx]);
                                            break;
                                        case 2:
                                            mcts_tree2.queue_from_main_thread_v.Enqueue(str_ret[idx]);
                                            break;
                                        case 3:
                                            mcts_tree3.queue_from_main_thread_v.Enqueue(str_ret[idx]);
                                            break;
                                        case 4:
                                            mcts_tree4.queue_from_main_thread_v.Enqueue(str_ret[idx]);
                                            break;
                                        case 5:
                                            mcts_tree5.queue_from_main_thread_v.Enqueue(str_ret[idx]);
                                            break;
                                    }
                                    idx += 1;
                                }
                                if (idx == v)
                                {
                                    break;
                                }
                            }
                            v = 0;
                            break;
                        }
                    }
                    flag = false;
                }
                if (checkMoves.Count > 0)
                {
                    str_mate_pv = "";
                    for (int i = 0; i < num_mate_tasks; i++)
                    {
                        switch (i)
                        {
                            case 0:
                                if (task_mate0.IsCompleted == true)
                                {
                                    if (mst0.is_mate_root == true)
                                    {
                                        str_mate_pv = mst0.root_str_pv;
                                        mate_first_move = mst0.first_move;
                                    }
                                }
                                break;
                            case 1:
                                if (task_mate1.IsCompleted == true)
                                {
                                    str_mate_pv = mst1.root_str_pv;
                                    mate_first_move = mst1.first_move;
                                }
                                break;
                        }
                    }
                }
                cnt = 0;
                for (int i = 0; i < num_tasks; i++)
                {
                    switch (i)
                    {
                        case 0:
                            if (task0.IsCompleted == true)
                            {
                                cnt += 1;
                            }
                            break;
                        case 1:
                            if (task1.IsCompleted == true)
                            {
                                cnt += 1;
                            }
                            break;
                        case 2:
                            if (task2.IsCompleted == true)
                            {
                                cnt += 1;
                            }
                            break;
                        case 3:
                            if (task3.IsCompleted == true)
                            {
                                cnt += 1;
                            }
                            break;
                        case 4:
                            if (task4.IsCompleted == true)
                            {
                                cnt += 1;
                            }
                            break;
                        case 5:
                            if (task5.IsCompleted == true)
                            {
                                cnt += 1;
                            }
                            break;
                    }
                }

                if (cnt == num_tasks)
                {
                    break;
                }
                index += 1;
                if (index == num_tasks)
                {
                    index = 0;
                }
            }

            if (str_mate_pv != "")
            {
                param_str_mate_pv = str_mate_pv;
                Console.WriteLine("詰みあり： " + str_mate_pv);
                //モンテカルロ木探索のタスクを終了させる。
                //はやく終了させるために、is_abortをTrueにしているが、
                //効果は今ひとつのようである。
                for (int i = 0; i < num_tasks; i++)
                {
                    switch (i)
                    {
                        case 0:
                            mcts_tree0.is_abort = true;
                            break;
                        case 1:
                            mcts_tree1.is_abort = true;
                            break;
                        case 2:
                            mcts_tree2.is_abort = true;
                            break;
                        case 3:
                            mcts_tree3.is_abort = true;
                            break;
                        case 4:
                            mcts_tree4.is_abort = true;
                            break;
                        case 5:
                            mcts_tree5.is_abort = true;
                            break;
                    }
                }
                cnt = 0;
                while (true)
                {
                    for (int i = 0; i < num_tasks; i++)
                    {
                        switch (i)
                        {
                            case 0:
                                if (task0.IsCompleted == true)
                                {
                                    cnt += 1;
                                }
                                break;
                            case 1:
                                if (task1.IsCompleted == true)
                                {
                                    cnt += 1;
                                }
                                break;
                            case 2:
                                if (task2.IsCompleted == true)
                                {
                                    cnt += 1;
                                }
                                break;
                            case 3:
                                if (task3.IsCompleted == true)
                                {
                                    cnt += 1;
                                }
                                break;
                            case 4:
                                if (task4.IsCompleted == true)
                                {
                                    cnt += 1;
                                }
                                break;
                            case 5:
                                if (task5.IsCompleted == true)
                                {
                                    cnt += 1;
                                }
                                break;
                        }
                    }
                    if (cnt == num_tasks)
                    {
                        break;
                    }
                }
                return;
            }
            List<float> win_rate_array = new List<float>();
            List<int> trial_count_array = new List<int>();
            List<MCTS4> temp_tree = new List<MCTS4>();
            for (int i = 0; i < num_tasks; i++)
            {
                switch(i)
                {
                    case 0:
                        temp_tree.Add(mcts_tree0);
                        break;
                    case 1:
                        temp_tree.Add(mcts_tree1);
                        break;
                    case 2:
                        temp_tree.Add(mcts_tree2);
                        break;
                    case 3:
                        temp_tree.Add(mcts_tree3);
                        break;
                    case 4:
                        temp_tree.Add(mcts_tree4);
                        break;
                    case 5:
                        temp_tree.Add(mcts_tree5);
                        break;
                }
            }
            List<Move> moves = mcts_tree0.TotalParam(num_tasks, ref win_rate_array, ref trial_count_array, ref temp_tree);
            str_first_move = Move2CSA(moves[0]);
            for (int i = 0; i < moves.Count; i++)
            {
                string sr = "";
                if (Move2CSA(moves[i]) == str_record_move)
                {
                    sr = "result= ○";
                    switch (i)
                    {
                        case 0:
                            move_first_accuracy[(int)bt.RootColor] += 1;
                            break;
                        case 1:
                            move_second_accuracy[(int)bt.RootColor] += 1;
                            break;
                        case 2:
                            move_third_accurasy[(int)bt.RootColor] += 1;
                            break;
                    }
                }
                else
                {
                    sr = "result= ×";
                }
                string str_color;
                if (bt.RootColor == Antares.Common.Color.Black)
                {
                    str_color = "+";
                }
                else
                {
                    str_color = "-";
                }
                string s = "候補手" + (i + 1).ToString() + ": " + str_color + Move2CSA(moves[i]) + ", " + sr + ", 勝率：" + win_rate_array[i].ToString("P2") + ", 訪問回数" + trial_count_array[i].ToString() + ", ";
                str_result = str_result + s;
                Console.WriteLine(s);
            }
        }

        public static void SearchWrapper2(ref BoardTree bt, int num_tasks, int num_mate_tasks, int thinking_time, int mate_search_depth, ref string str_result, ref string param_str_mate_pv, string str_record_move,
                              ref int[] move_first_accuracy, ref int[] move_second_accuracy, ref int[] move_third_accurasy, ref Move mate_first_move, ref string str_first_move)
        {
            MCTS5 mcts_tree0 = new MCTS5();
            MCTS5 mcts_tree1 = new MCTS5();
            MCTS5 mcts_tree2 = new MCTS5();
            MCTS5 mcts_tree3 = new MCTS5();
            MCTS5 mcts_tree4 = new MCTS5();
            MCTS5 mcts_tree5 = new MCTS5();
            string str_mate_pv = "";

            for (int i = 0; i < num_tasks; i++)
            {
                switch (i)
                {
                    case 0:
                        mcts_tree0 = new MCTS5();
                        mcts_tree0.TaskNumber = 0;
                        mcts_tree0.BTree = DeepCopy(bt, false);
                        mcts_tree0.GenRootMoves();
                        mcts_tree0.SearchTimeLimit = thinking_time * 1000;
                        mcts_tree0.SetRootOutput();
                        break;
                    case 1:
                        mcts_tree1 = new MCTS5();
                        mcts_tree1.TaskNumber = 1;
                        mcts_tree1.BTree = DeepCopy(bt, false);
                        mcts_tree1.GenRootMoves();
                        mcts_tree1.SearchTimeLimit = thinking_time * 1000;
                        mcts_tree1.SetRootOutput();
                        break;
                    case 2:
                        mcts_tree2 = new MCTS5();
                        mcts_tree2.TaskNumber = 2;
                        mcts_tree2.BTree = DeepCopy(bt, false);
                        mcts_tree2.GenRootMoves();
                        mcts_tree2.SearchTimeLimit = thinking_time * 1000;
                        mcts_tree2.SetRootOutput();
                        break;
                    case 3:
                        mcts_tree3 = new MCTS5();
                        mcts_tree3.TaskNumber = 3;
                        mcts_tree3.BTree = DeepCopy(bt, false);
                        mcts_tree3.GenRootMoves();
                        mcts_tree3.SearchTimeLimit = thinking_time * 1000;
                        mcts_tree3.SetRootOutput();
                        break;
                    case 4:
                        mcts_tree4 = new MCTS5();
                        mcts_tree4.TaskNumber = 4;
                        mcts_tree4.BTree = DeepCopy(bt, false);
                        mcts_tree4.GenRootMoves();
                        mcts_tree4.SearchTimeLimit = thinking_time * 1000;
                        mcts_tree4.SetRootOutput();
                        break;
                    case 5:
                        mcts_tree5 = new MCTS5();
                        mcts_tree5.TaskNumber = 5;
                        mcts_tree5.BTree = DeepCopy(bt, false);
                        mcts_tree5.GenRootMoves();
                        mcts_tree5.SearchTimeLimit = thinking_time * 1000;
                        mcts_tree5.SetRootOutput();
                        break;
                }
            }
            List<Move> checkMoves = new List<Move>();
            List<Move> li_checkMoves0 = new List<Move>();
            List<Move> li_checkMoves1 = new List<Move>();
            Mate mst0 = new Mate();
            Mate mst1 = new Mate();

            GenCheck(bt, (int)bt.RootColor, ref checkMoves);
            if (checkMoves.Count > 0)
            {
                for (int i = 0; i < checkMoves.Count; i++)
                {
                    if (i % 2 == 0)
                    {
                        li_checkMoves0.Add(checkMoves[i]);
                    }
                    else
                    {
                        li_checkMoves1.Add(checkMoves[i]);
                    }
                }
                for (int i = 0; i < num_mate_tasks; i++)
                {
                    switch (i)
                    {
                        case 0:
                            mst0.BTree = DeepCopy(bt, false);
                            mst0.RootCheckMoves = new List<Move>(li_checkMoves0);
                            mst0.max_ply = mate_search_depth;
                            break;
                        case 1:
                            mst1.BTree = DeepCopy(bt, false);
                            mst1.RootCheckMoves = new List<Move>(li_checkMoves1);
                            mst1.max_ply = mate_search_depth;
                            break;
                    }
                }
            }

            Task task0 = new Task(() => mcts_tree0.Root());
            Task task1 = new Task(() => mcts_tree1.Root());
            Task task2 = new Task(() => mcts_tree2.Root());
            Task task3 = new Task(() => mcts_tree3.Root()); ;
            Task task4 = new Task(() => mcts_tree4.Root()); ;
            Task task5 = new Task(() => mcts_tree5.Root());
            for (int i = 0; i < num_tasks; i++)
            {
                switch (i)
                {
                    case 0:
                        //task0 = Task.Run(() => mcts4.mcts_tree0.Root());
                        task0.Start();
                        break;
                    case 1:
                        //task1 = Task.Run(() => mcts4.mcts_tree1.Root());
                        task1.Start();
                        break;
                    case 2:
                        //task2 = Task.Run(() => mcts4.mcts_tree2.Root());
                        task2.Start();
                        break;
                    case 3:
                        //task3 = Task.Run(() => mcts4.mcts_tree3.Root());
                        task3.Start();
                        break;
                    case 4:
                        //task4 = Task.Run(() => mcts4.mcts_tree4.Root());
                        task4.Start();
                        break;
                    case 5:
                        //task5 = Task.Run(() => mcts4.mcts_tree5.Root());
                        task5.Start();
                        break;
                }
            }
            Task task_mate0 = new Task(() => mst0.MateSearchWrapper(mate_search_depth));
            Task task_mate1 = new Task(() => mst1.MateSearchWrapper(mate_search_depth));
            if (checkMoves.Count > 0)
            {
                for (int i = 0; i < num_mate_tasks; i++)
                {
                    switch (i)
                    {
                        case 0:
                            //task_mate0 = Task.Run(() => mst0.MateSearchWrapper());
                            task_mate0.Start();
                            break;
                        case 1:
                            //task_mate1 = Task.Run(() => mst1.MateSearchWrapper());
                            task_mate1.Start();
                            break;
                    }
                }
            }

            int index = 0;
            string[] str_policy_requests = new string[num_tasks];
            string[] str_value_requests = new string[num_tasks];
            for (int i = 0; i < num_tasks; i++)
            {
                str_policy_requests[i] = "";
                str_value_requests[i] = "";
            }
            string temp_s;
            string[] temp_s2;
            bool flag = false;
            //bool is_completed = false;
            int p = 0;
            int v = 0;
            int cnt = 0;
            Stopwatch sw = new Stopwatch();
            sw.Start();
            //int cnt_0 = 0;
            //int cnt_1 = 0;
            long base_time = sw.ElapsedMilliseconds;
            while (true)
            {
                switch (index)
                {
                    case 0:
                        if (mcts_tree0.queue_to_main_thread_p.Count > 0)
                        {
                            temp_s = mcts_tree0.queue_to_main_thread_p.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_policy_requests[index] = temp_s2[1];
                            p += 1;
                        }
                        else if (mcts_tree0.queue_to_main_thread_v.Count > 0)
                        {
                            temp_s = mcts_tree0.queue_to_main_thread_v.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_value_requests[index] = temp_s2[1];
                            v += 1;
                        }
                        break;
                    case 1:
                        if (mcts_tree1.queue_to_main_thread_p.Count > 0)
                        {
                            temp_s = mcts_tree1.queue_to_main_thread_p.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_policy_requests[index] = temp_s2[1];
                            p += 1;
                        }
                        else if (mcts_tree1.queue_to_main_thread_v.Count > 0)
                        {
                            temp_s = mcts_tree1.queue_to_main_thread_v.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_value_requests[index] = temp_s2[1];
                            v += 1;
                        }
                        break;
                    case 2:
                        if (mcts_tree2.queue_to_main_thread_p.Count > 0)
                        {
                            temp_s = mcts_tree2.queue_to_main_thread_p.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_policy_requests[index] = temp_s2[1];
                            p += 1;
                        }
                        else if (mcts_tree2.queue_to_main_thread_v.Count > 0)
                        {
                            temp_s = mcts_tree2.queue_to_main_thread_v.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_value_requests[index] = temp_s2[1];
                            v += 1;
                        }
                        break;
                    case 3:
                        if (mcts_tree3.queue_to_main_thread_p.Count > 0)
                        {
                            temp_s = mcts_tree3.queue_to_main_thread_p.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_policy_requests[index] = temp_s2[1];
                            p += 1;
                        }
                        else if (mcts_tree3.queue_to_main_thread_v.Count > 0)
                        {
                            temp_s = mcts_tree3.queue_to_main_thread_v.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_value_requests[index] = temp_s2[1];
                            v += 1;
                        }
                        break;
                    case 4:
                        if (mcts_tree4.queue_to_main_thread_p.Count > 0)
                        {
                            temp_s = mcts_tree4.queue_to_main_thread_p.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_policy_requests[index] = temp_s2[1];
                            p += 1;
                        }
                        else if (mcts_tree4.queue_to_main_thread_v.Count > 0)
                        {
                            temp_s = mcts_tree4.queue_to_main_thread_v.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_value_requests[index] = temp_s2[1];
                            v += 1;
                        }
                        break;
                    case 5:
                        if (mcts_tree5.queue_to_main_thread_p.Count > 0)
                        {
                            temp_s = mcts_tree5.queue_to_main_thread_p.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_policy_requests[index] = temp_s2[1];
                            p += 1;
                        }
                        else if (mcts_tree5.queue_to_main_thread_v.Count > 0)
                        {
                            temp_s = mcts_tree5.queue_to_main_thread_v.Dequeue();
                            temp_s2 = temp_s.Split(",");
                            str_value_requests[index] = temp_s2[1];
                            v += 1;
                        }
                        break;
                }
                if (v == num_tasks || p == num_tasks)
                {
                    flag = true;
                }
                //制限時間に達したらis_abortをTrueにする。
                long elapsed = sw.ElapsedMilliseconds;
                if (elapsed - base_time > 300)
                {
                    base_time = elapsed;
                    flag = true;
                }

                if (elapsed > thinking_time * 1000)
                {
                    for (int i = 0; i < num_tasks; i++)
                    {
                        switch (i)
                        {
                            case 0:
                                mcts_tree0.is_abort = true;
                                break;
                            case 1:
                                mcts_tree1.is_abort = true;
                                break;
                            case 2:
                                mcts_tree2.is_abort = true;
                                break;
                            case 3:
                                mcts_tree3.is_abort = true;
                                break;
                            case 4:
                                mcts_tree4.is_abort = true;
                                break;
                            case 5:
                                mcts_tree5.is_abort = true;
                                break;
                        }
                    }
                }

                if (flag == true)
                {
                    while (true)
                    {
                        if (p == 0 && v == 0)
                        {
                            break;
                        }
                        if (p > 0)
                        {
                            string[] str_sfen = new string[p];
                            int idx = 0;
                            for (int i = 0; i < p; i++)
                            {
                                str_sfen[i] = "";
                            }
                            for (int i = 0; i < num_tasks; i++)
                            {
                                if (str_policy_requests[i] != "")
                                {
                                    str_sfen[idx] = str_policy_requests[i];
                                    idx += 1;
                                }
                                if (idx == p)
                                {
                                    break;
                                }
                            }
                            string[] str_ret = ExecPolicy(str_sfen);
                            idx = 0;
                            for (int i = 0; i < num_tasks; i++)
                            {
                                if (str_policy_requests[i] != "")
                                {


                                    switch (i)
                                    {
                                        case 0:
                                            mcts_tree0.queue_from_main_thread_p.Enqueue(str_ret[idx]);
                                            break;
                                        case 1:
                                            mcts_tree1.queue_from_main_thread_p.Enqueue(str_ret[idx]);
                                            break;
                                        case 2:
                                            mcts_tree2.queue_from_main_thread_p.Enqueue(str_ret[idx]);
                                            break;
                                        case 3:
                                            mcts_tree3.queue_from_main_thread_p.Enqueue(str_ret[idx]);
                                            break;
                                        case 4:
                                            mcts_tree4.queue_from_main_thread_p.Enqueue(str_ret[idx]);
                                            break;
                                        case 5:
                                            mcts_tree5.queue_from_main_thread_p.Enqueue(str_ret[idx]);
                                            break;
                                    }
                                    idx += 1;
                                }
                                if (idx == p)
                                {
                                    break;
                                }
                            }
                            p = 0;
                            break;
                        }
                        if (v > 0)
                        {
                            string[] str_sfen = new string[v];
                            int idx = 0;
                            for (int i = 0; i < v; i++)
                            {
                                str_sfen[i] = "";
                            }
                            for (int i = 0; i < num_tasks; i++)
                            {
                                if (str_value_requests[i] != "")
                                {
                                    str_sfen[idx] = str_value_requests[i];
                                    idx += 1;
                                }
                                if (idx == v)
                                {
                                    break;
                                }
                            }
                            string[] str_ret;
                            try
                            {
                                str_ret = ExecValue(str_sfen);
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine("例外が発生しましたが、処理を続行します。");
                                int temp_l = str_sfen.Length;
                                str_ret = new string[temp_l];
                                for (int j = 0; j < temp_l; j++)
                                {
                                    str_ret[j] = "0.0";
                                }
                            }
                            idx = 0;
                            for (int i = 0; i < num_tasks; i++)
                            {
                                if (str_value_requests[i] != "")
                                {
                                    switch (i)
                                    {
                                        case 0:
                                            mcts_tree0.queue_from_main_thread_v.Enqueue(str_ret[idx]);
                                            break;
                                        case 1:
                                            mcts_tree1.queue_from_main_thread_v.Enqueue(str_ret[idx]);
                                            break;
                                        case 2:
                                            mcts_tree2.queue_from_main_thread_v.Enqueue(str_ret[idx]);
                                            break;
                                        case 3:
                                            mcts_tree3.queue_from_main_thread_v.Enqueue(str_ret[idx]);
                                            break;
                                        case 4:
                                            mcts_tree4.queue_from_main_thread_v.Enqueue(str_ret[idx]);
                                            break;
                                        case 5:
                                            mcts_tree5.queue_from_main_thread_v.Enqueue(str_ret[idx]);
                                            break;
                                    }
                                    idx += 1;
                                }
                                if (idx == v)
                                {
                                    break;
                                }
                            }
                            v = 0;
                            break;
                        }
                    }
                    flag = false;
                }
                if (checkMoves.Count > 0)
                {
                    str_mate_pv = "";
                    for (int i = 0; i < num_mate_tasks; i++)
                    {
                        switch (i)
                        {
                            case 0:
                                if (task_mate0.IsCompleted == true)
                                {
                                    if (mst0.is_mate_root == true)
                                    {
                                        str_mate_pv = mst0.root_str_pv;
                                        mate_first_move = mst0.first_move;
                                    }
                                }
                                break;
                            case 1:
                                if (task_mate1.IsCompleted == true)
                                {
                                    str_mate_pv = mst1.root_str_pv;
                                    mate_first_move = mst1.first_move;
                                }
                                break;
                        }
                    }
                }
                cnt = 0;
                for (int i = 0; i < num_tasks; i++)
                {
                    switch (i)
                    {
                        case 0:
                            if (task0.IsCompleted == true)
                            {
                                cnt += 1;
                            }
                            break;
                        case 1:
                            if (task1.IsCompleted == true)
                            {
                                cnt += 1;
                            }
                            break;
                        case 2:
                            if (task2.IsCompleted == true)
                            {
                                cnt += 1;
                            }
                            break;
                        case 3:
                            if (task3.IsCompleted == true)
                            {
                                cnt += 1;
                            }
                            break;
                        case 4:
                            if (task4.IsCompleted == true)
                            {
                                cnt += 1;
                            }
                            break;
                        case 5:
                            if (task5.IsCompleted == true)
                            {
                                cnt += 1;
                            }
                            break;
                    }
                }

                if (cnt == num_tasks)
                {
                    break;
                }
                index += 1;
                if (index == num_tasks)
                {
                    index = 0;
                }
            }

            if (str_mate_pv != "")
            {
                param_str_mate_pv = str_mate_pv;
                Console.WriteLine("詰みあり： " + str_mate_pv);
                //モンテカルロ木探索のタスクを終了させる。
                //はやく終了させるために、is_abortをTrueにしているが、
                //効果は今ひとつのようである。
                for (int i = 0; i < num_tasks; i++)
                {
                    switch (i)
                    {
                        case 0:
                            mcts_tree0.is_abort = true;
                            break;
                        case 1:
                            mcts_tree1.is_abort = true;
                            break;
                        case 2:
                            mcts_tree2.is_abort = true;
                            break;
                        case 3:
                            mcts_tree3.is_abort = true;
                            break;
                        case 4:
                            mcts_tree4.is_abort = true;
                            break;
                        case 5:
                            mcts_tree5.is_abort = true;
                            break;
                    }
                }
                cnt = 0;
                while (true)
                {
                    for (int i = 0; i < num_tasks; i++)
                    {
                        switch (i)
                        {
                            case 0:
                                if (task0.IsCompleted == true)
                                {
                                    cnt += 1;
                                }
                                break;
                            case 1:
                                if (task1.IsCompleted == true)
                                {
                                    cnt += 1;
                                }
                                break;
                            case 2:
                                if (task2.IsCompleted == true)
                                {
                                    cnt += 1;
                                }
                                break;
                            case 3:
                                if (task3.IsCompleted == true)
                                {
                                    cnt += 1;
                                }
                                break;
                            case 4:
                                if (task4.IsCompleted == true)
                                {
                                    cnt += 1;
                                }
                                break;
                            case 5:
                                if (task5.IsCompleted == true)
                                {
                                    cnt += 1;
                                }
                                break;
                        }
                    }
                    if (cnt == num_tasks)
                    {
                        break;
                    }
                }
                return;
            }
            List<float> win_rate_array = new List<float>();
            List<int> trial_count_array = new List<int>();
            List<MCTS5> temp_tree = new List<MCTS5>();
            for (int i = 0; i < num_tasks; i++)
            {
                switch (i)
                {
                    case 0:
                        temp_tree.Add(mcts_tree0);
                        break;
                    case 1:
                        temp_tree.Add(mcts_tree1);
                        break;
                    case 2:
                        temp_tree.Add(mcts_tree2);
                        break;
                    case 3:
                        temp_tree.Add(mcts_tree3);
                        break;
                    case 4:
                        temp_tree.Add(mcts_tree4);
                        break;
                    case 5:
                        temp_tree.Add(mcts_tree5);
                        break;
                }
            }
            List<Move> moves = mcts_tree0.TotalParam(num_tasks, ref win_rate_array, ref trial_count_array, ref temp_tree);
            str_first_move = Move2CSA(moves[0]);
            for (int i = 0; i < moves.Count; i++)
            {
                string sr = "";
                if (Move2CSA(moves[i]) == str_record_move)
                {
                    sr = "result= ○";
                    switch (i)
                    {
                        case 0:
                            move_first_accuracy[(int)bt.RootColor] += 1;
                            break;
                        case 1:
                            move_second_accuracy[(int)bt.RootColor] += 1;
                            break;
                        case 2:
                            move_third_accurasy[(int)bt.RootColor] += 1;
                            break;
                    }
                }
                else
                {
                    sr = "result= ×";
                }
                string str_color;
                if (bt.RootColor == Antares.Common.Color.Black)
                {
                    str_color = "+";
                }
                else
                {
                    str_color = "-";
                }
                string s = "候補手" + (i + 1).ToString() + ": " + str_color + Move2CSA(moves[i]) + ", " + sr + ", 勝率：" + win_rate_array[i].ToString("P2") + ", 訪問回数" + trial_count_array[i].ToString() + ", ";
                str_result = str_result + s;
                Console.WriteLine(s);
            }
        }


        public static void AnalyzeRandomPlayout(string str_sfen, int playout_limit)
        {
            const int num_threads = 12;
            int color;
            int i, j;

            StreamWriter sw = IO.OpenStreamWriter("log_play_out_multi_task.txt");

            BoardTree bt = new BoardTree();
            Board.BoardTreeAlloc(ref bt);
            Board.Init(ref bt);
            bt = SFEN.ToBoard(str_sfen);
            color = (int)bt.RootColor;
            List<Move> moves = new List<Move>();
            if (IsAttacked(bt, bt.SQ_King[color], color) == 0)
            {
                GenDrop(bt, color, ref moves);
                GenNoCap(bt, color, ref moves);
                GenCap(bt, color, ref moves);
            }
            else
            {
                GenEvasion(bt, color, ref moves);
            }

            for (i = 0; i < moves.Count; i++)
            {
                Move move = moves[i];
                Do(ref bt, move, color);
                if (IsAttacked(bt, bt.SQ_King[color], color) > 0)
                    moves.RemoveAt(i);
                UnDo(ref bt, move, color);
            }

            int move_count = moves.Count;

            List<Move> split_moves0 = new List<Move>();
            List<Move> split_moves1 = new List<Move>();
            List<Move> split_moves2 = new List<Move>();
            List<Move> split_moves3 = new List<Move>();
            List<Move> split_moves4 = new List<Move>();
            List<Move> split_moves5 = new List<Move>();
            List<Move> split_moves6 = new List<Move>();
            List<Move> split_moves7 = new List<Move>();
            List<Move> split_moves8 = new List<Move>();
            List<Move> split_moves9 = new List<Move>();
            List<Move> split_moves10 = new List<Move>();
            List<Move> split_moves11 = new List<Move>();

            int task_number = 0;
            for (i = 0; i < move_count; i++)
            {
                switch (task_number)
                {
                    case 0:
                        split_moves0.Add(moves[i]);
                        break;
                    case 1:
                        split_moves1.Add(moves[i]);
                        break;
                    case 2:
                        split_moves2.Add(moves[i]);
                        break;
                    case 3:
                        split_moves3.Add(moves[i]);
                        break;
                    case 4:
                        split_moves4.Add(moves[i]);
                        break;
                    case 5:
                        split_moves5.Add(moves[i]);
                        break;
                    case 6:
                        split_moves6.Add(moves[i]);
                        break;
                    case 7:
                        split_moves7.Add(moves[i]);
                        break;
                    case 8:
                        split_moves8.Add(moves[i]);
                        break;
                    case 9:
                        split_moves9.Add(moves[i]);
                        break;
                    case 10:
                        split_moves10.Add(moves[i]);
                        break;
                    case 11:
                        split_moves11.Add(moves[i]);
                        break;
                }
                ++task_number;
                if (task_number == num_threads)
                {
                    task_number = 0;
                }
            }

            Tree tree0 = new Tree();
            Tree tree1 = new Tree();
            Tree tree2 = new Tree();
            Tree tree3 = new Tree();
            Tree tree4 = new Tree();
            Tree tree5 = new Tree();
            Tree tree6 = new Tree();
            Tree tree7 = new Tree();
            Tree tree8 = new Tree();
            Tree tree9 = new Tree();
            Tree tree10 = new Tree();
            Tree tree11 = new Tree();

            Task[] tasks = new Task[num_threads];

            BoardTree bt0 = Board.DeepCopy(bt, false);
            tasks[0] = Task.Factory.StartNew(() => PlayOutLoop(ref bt0, color, split_moves0, split_moves0.Count, ref tree0, playout_limit));
            BoardTree bt1 = Board.DeepCopy(bt, false);
            tasks[1] = Task.Factory.StartNew(() => PlayOutLoop(ref bt1, color, split_moves1, split_moves1.Count, ref tree1, playout_limit));
            BoardTree bt2 = Board.DeepCopy(bt, false);
            tasks[2] = Task.Factory.StartNew(() => PlayOutLoop(ref bt2, color, split_moves2, split_moves2.Count, ref tree2, playout_limit));
            BoardTree bt3 = Board.DeepCopy(bt, false);
            tasks[3] = Task.Factory.StartNew(() => PlayOutLoop(ref bt3, color, split_moves3, split_moves3.Count, ref tree3, playout_limit));
            BoardTree bt4 = Board.DeepCopy(bt, false);
            tasks[4] = Task.Factory.StartNew(() => PlayOutLoop(ref bt4, color, split_moves4, split_moves4.Count, ref tree4, playout_limit));
            BoardTree bt5 = Board.DeepCopy(bt, false);
            tasks[5] = Task.Factory.StartNew(() => PlayOutLoop(ref bt5, color, split_moves5, split_moves5.Count, ref tree5, playout_limit));
            BoardTree bt6 = Board.DeepCopy(bt, false);
            tasks[6] = Task.Factory.StartNew(() => PlayOutLoop(ref bt6, color, split_moves6, split_moves6.Count, ref tree6, playout_limit));
            BoardTree bt7 = Board.DeepCopy(bt, false);
            tasks[7] = Task.Factory.StartNew(() => PlayOutLoop(ref bt7, color, split_moves7, split_moves7.Count, ref tree7, playout_limit));
            BoardTree bt8 = Board.DeepCopy(bt, false);
            tasks[8] = Task.Factory.StartNew(() => PlayOutLoop(ref bt8, color, split_moves8, split_moves8.Count, ref tree8, playout_limit));
            BoardTree bt9 = Board.DeepCopy(bt, false);
            tasks[9] = Task.Factory.StartNew(() => PlayOutLoop(ref bt9, color, split_moves9, split_moves9.Count, ref tree9, playout_limit));
            BoardTree bt10 = Board.DeepCopy(bt, false);
            tasks[10] = Task.Factory.StartNew(() => PlayOutLoop(ref bt10, color, split_moves10, split_moves10.Count, ref tree10, playout_limit));
            BoardTree bt11 = Board.DeepCopy(bt, false);
            tasks[11] = Task.Factory.StartNew(() => PlayOutLoop(ref bt11, color, split_moves11, split_moves11.Count, ref tree11, playout_limit));
            Task.WaitAll(tasks[0]);
            Task.WaitAll(tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[6], tasks[7], tasks[8], tasks[9], tasks[10], tasks[11]);

            List<Tree> trees = new List<Tree>();
            trees.Add(tree0);
            trees.Add(tree1);
            trees.Add(tree2);
            trees.Add(tree3);
            trees.Add(tree4);
            trees.Add(tree5);
            trees.Add(tree6);
            trees.Add(tree7);
            trees.Add(tree8);
            trees.Add(tree9);
            trees.Add(tree10);
            trees.Add(tree11);

            List<List<Move>> moves_list = new List<List<Move>>();
            moves_list.Add(split_moves0);
            moves_list.Add(split_moves1);
            moves_list.Add(split_moves2);
            moves_list.Add(split_moves3);
            moves_list.Add(split_moves4);
            moves_list.Add(split_moves5);
            moves_list.Add(split_moves6);
            moves_list.Add(split_moves7);
            moves_list.Add(split_moves8);
            moves_list.Add(split_moves9);
            moves_list.Add(split_moves10);
            moves_list.Add(split_moves11);

            i = 0;
            List<float> win_rates = new List<float>();
            List<Move> best_moves_list = new List<Move>();
            Move[] best_moves = new Move[3];
            foreach (var tree in trees)
            {
                List<Move> temp_moves_list = moves_list[i];
                j = 0;
                foreach (var move in temp_moves_list)
                {
                    string str_move = Move2CSA(move);
                    float win_rate = (float)(tree.list_win[j] + tree.list_draw[j] * 0.5) / (tree.list_win[j] + tree.list_lose[j] + tree.list_draw[j]);
                    win_rates.Add(win_rate);
                    best_moves_list.Add(move);
                    Console.WriteLine(str_move + ", win_rate=" + win_rate.ToString("F2") +
                        ", win_count=" + tree.list_win[j].ToString() +
                        ", lose_count=" + tree.list_lose[j].ToString() +
                        ", draw_count=" + tree.list_draw[j].ToString());
                    j++;
                    /*Console.WriteLine("thread=" + i.ToString() + ", move=" + str_move + ", win_count=" + tree.list_win[i].ToString() +
                        ", lose_count=" + tree.list_lose[i].ToString() + ", draw_count=" + tree.list_draw[i].ToString());*/
                }
                i++;
            }

            float win_rate_1st = new float();
            float win_rate_2nd = new float();
            float win_rate_3rd = new float();

            win_rate_1st = win_rates.Max();
            int index_1st = win_rates.IndexOf(win_rate_1st);
            best_moves[0] = best_moves_list[index_1st];
            win_rates[index_1st] = -1.0f; // 1位を除外
            win_rate_2nd = win_rates.Max();
            int index_2nd = win_rates.IndexOf(win_rate_2nd);
            best_moves[1] = best_moves_list[index_2nd];
            win_rates[index_2nd] = -1.0f; // 2位を除外
            win_rate_3rd = win_rates.Max();
            int index_3rd = win_rates.IndexOf(win_rate_3rd);
            best_moves[2] = best_moves_list[index_3rd];

            Console.WriteLine();

            string str_out = "";
            //str_out += "ply=" + (counter + 1).ToString() + ", ";
            //str_out += "best_move_1st=" + Move2CSA(best_moves[0]) + ", win_rate=" + win_rate_1st.ToString("F2") + Environment.NewLine;
            //str_out += "best_move_2nd=" + Move2CSA(best_moves[1]) + ", win_rate=" + win_rate_2nd.ToString("F2") + Environment.NewLine;
            //str_out += "best_move_3rd=" + Move2CSA(best_moves[2]) + ", win_rate=" + win_rate_3rd.ToString("F2") + Environment.NewLine;
            str_out += "best_move_1st=" + Move2CSA(best_moves[0]) + ", win_rate=" + win_rate_1st.ToString("F2") + ", ";
            str_out += "best_move_2nd=" + Move2CSA(best_moves[1]) + ", win_rate=" + win_rate_2nd.ToString("F2") + ", ";
            str_out += "best_move_3rd=" + Move2CSA(best_moves[2]) + ", win_rate=" + win_rate_3rd.ToString("F2");
            sw.WriteLine(str_out);
            sw.Close();
            Console.WriteLine(str_out);
        }

        public static string IsMate(string str_sfen, int depth_limit)
        {
            Init();
            BoardTree bt = new BoardTree();
            BoardTreeAlloc(ref bt);
            Board.Init(ref bt);
            const int root_move_limit = 128;
            //string str_sfen = "6s2/6R2/6Bk1/6p2/7N1/9/9/9/9 b GN 1";
            //string str_sfen = "9/9/9/9/1n7/2P6/1Kb6/2r6/2S6 w gn 1";
            //string str_sfen = "6k2/5p1r1/5BN2/9/9/9/9/9/9 b BN 1";
            //string str_sfen = "5g2l/7s1/5B1kb/6R2/6P2/9/9/9/9 b G 1";
            //string str_sfen = "5sknl/3R5/4p1bp1/9/9/9/9/9/9 b RBS 1";
            //string str_sfen = "9/4+RB1k1/5+r1pp/5b3/9/9/9/9/9 b 2G 1";
            //string str_sfen = "7k1/4p3r/6S2/9/6rN1/9/9/9/9 b BNL 1"; // 13手詰め
            //string str_sfen = "6skB/4g3l/8p/5p3/5N3/9/9/9/9 b GS 1";
            //string str_sfen = "9/5gsk1/6n2/6R2/5B1Pp/9/9/9/9 b BG 1";
            //string str_sfen = "9/9/9/9/9/P1p6/1S1p5/9/rNK6 w gs 1";
            bt = SFEN.ToBoard(str_sfen);
            List<Move> pv = new List<Move>();
            int rest_depth = depth_limit;
            int i, j, k;
            Mate mate = new Mate();

            mate.max_ply = rest_depth;
            mate.move_cur = new Move[root_move_limit];

            for (i = 0; i < root_move_limit; i++)
            {
                mate.move_cur[i] = new Move();
            }

            List<Move> checkMoves = mate.GenRootCheckMoves(ref bt);
            mate.RootCheckMoves = checkMoves;
            string str_pv = "";

            if (mate.Offend(ref bt, (int)bt.RootColor, rest_depth, 1))
            {
                List<List<Move>> l = mate.mate_proc;
                List<List<Move>> nl = mate.no_mate_proc;
                Move fm = mate.first_move;
                Move sm = mate.second_move;
                int index = 0;

                for (i = 0; i < l.Count; i++)
                {
                    if (l[i][0].Value == fm.Value && l[i][1].Value == sm.Value)
                    {
                        index = i;
                        break;
                    }
                }

                bool b = false;
                List<int> idxes = new List<int>();
                //int a = 0;
                for (i = 0; i < l.Count; i++)
                {
                    string s = i.ToString() + " / " + l.Count.ToString();
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
                            else
                            {
                                int z = 0;
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
                    if (idxes.Contains(i))
                        continue;
                    str_pv = "";
                    for (j = 0; j < rest_depth; j++)
                    {
                        //str_pv += CSA.Move2CSA(l[index][i]);
                        str_pv += CSA.Move2CSA(l[i][j]);
                        if (j != rest_depth - 1)
                            str_pv += ", ";
                    }
                    Console.WriteLine(str_pv);
                }
            }
            return str_pv;
        }

        public static void RootLoop(int thread_id, ref AlphaBetaTree abt, Sort.MoveAndScore[,] mas_after, List<Move> RootMoves, int ith, ref Move best_move, ref int best_value, int temp_best_value, List<int> extensions)
        {
            int value, ifrom, ito, icap_pc, prev_pv_length, alpha, beta, alpha_old;
            uint state_node_new;
            List<Move> local_pv = new List<Move>();
            List<List<Move>> backup_pvs = new List<List<Move>>();
            Direction idirec;
            int ply = 1;
            int color = (int)abt.bt.RootColor;
            best_move = new Move();
            value = best_value = 0;
            int i = abt.task_number;
            int iteration = 1;
            prev_pv_length = 0;
            //alpha = Value_Min;
            //beta = Value_Max;
            //alpha_old = alpha;
            while (iteration <= ith)
            {
                local_pv.Clear();
                alpha = Value_Min;
                beta = Value_Max;
                alpha_old = alpha;
                //int score_first = -mas_after[i, 0].score;// ToDo: ここで最初の手のスコアから、すべての兄弟手の最高評価の手に変更する。
                int score_first = -temp_best_value;
                int score_last = -abt.mas_after[ply, RootMoves.Count - 1].score;
                bool is_stable = true;
                if (score_first - score_last < 512)
                {
                    is_stable = false;
                }

                for (int j = 0; j < RootMoves.Count; j++)
                {
                    int current_score = -mas_after[i, j].score;

                    // Futility Pruningもどき
                    //if (i > 0 && score_diff >= 768)
                    int score_diff = score_first - current_score;
                    if (j > 0 && score_diff >= futility_margin)
                    {
                        abt.futility_cut++;
                        break;
                    }
                    /*int score_diff_prev = prev_value - current_score;

                    if (i > 0 && score_diff_prev >= 768)
                    {
                        abt.futility_cut++;
                        break;
                    }*/

                    abt.bt.EvalArray[ply] = -mas_after[i, j].score;

                    Move current_move = mas_after[i, j].move;
                    abt.current_move[ply] = current_move;
                    //abt.current_move[ply] = moves[i];

                    // 1手進める。
                    Do(ref abt.bt, current_move, color);

                    // 自玉がDiscovered Checkになってしまった場合
                    if (IsAttacked(abt.bt, abt.bt.SQ_King[color], color) > 0)
                    {
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
                        abt.is_check[ply] = true;

                    int extension = 0;
                    extension = extensions[j];

                    // 【重要】：ここにバグあり。
                    // ルートの手を分割する前のインデックスで延長を割り当てないといけない。
                    // 後で変更する。
                    /*if (extensions[j] == 0)
                        extension += Ply_Inc / 2;
                    if (extensions[j] == 1)
                        extension += Ply_Inc / 4;
                    if (extensions[j] == 2)
                        extension += Ply_Inc / 8;
                    if (is_stable && extensions[j] >= 12)
                        extension -= Ply_Inc / 4;
                    if (!is_stable && extensions[j] >= 16)
                        extension -= Ply_Inc / 4;*/

                    state_node_new = (uint)AlphaBetaTree.NodeState.node_pv | (uint)AlphaBetaTree.NodeState.node_do_null_move
                        | (uint)AlphaBetaTree.NodeState.node_do_delta | (uint)AlphaBetaTree.NodeState.node_do_razoring
                        | (uint)AlphaBetaTree.NodeState.node_do_futility | (uint)AlphaBetaTree.NodeState.node_do_probcut;


                    List<Move> temp_pv = new List<Move>();

                    // 次のノードを展開する。
                    value = -Search(ref abt, color ^ 1, -beta, -alpha, iteration * Ply_Inc + extension, ply + 1, state_node_new, -mas_after[i, j].score, ref temp_pv);// ※残り深さとstate_nodeは後で変更する。

                    // 1手戻す。
                    UnDo(ref abt.bt, current_move, color);

                    alpha = Math.Max(alpha, value);

                    if (alpha >= beta)
                    {
                        break;
                    }

                    // ToDo: おそらくここでPVを更新する必要がある。
                    if (j == 0)
                    {
                        //abt.bt.EvalArray[ply] = alpha;
                        best_value = value;
                        best_move = current_move;
                        for (j = 0; j < temp_pv.Count; j++)
                        {
                            local_pv.Insert(0, temp_pv[j]);
                        }
                        local_pv.Insert(0, current_move);
                        prev_pv_length = temp_pv.Count + 1;
                        abt.bt.EvalArray[ply] = best_value;
                        abt.tt.Store(abt.bt.CurrentHash, value, (Common.Color)color, false, abt.current_move[ply], ply);
                    }
                    else
                    {
                        if (alpha != alpha_old)
                        {
                            if (temp_pv.Count >= prev_pv_length)
                            {
                                best_value = value;
                                local_pv.Clear();
                                for (j = 0; j < temp_pv.Count; j++)
                                {
                                    local_pv.Insert(0, temp_pv[j]);
                                }
                                local_pv.Insert(0, current_move);
                                prev_pv_length = temp_pv.Count + 1;
                                abt.bt.EvalArray[ply] = alpha;
                                abt.tt.Store(abt.bt.CurrentHash, value, (Common.Color)color, false, abt.current_move[ply], ply);
                            }
                            else
                            {
                                alpha = alpha_old;
                            }
                        }
                    }
                    alpha_old = alpha;
                }
                backup_pvs.Add(local_pv.ToList());
                iteration++;
            }

            int max_pv_length = 0;
            for (i = 0; i < backup_pvs.Count; i++)
            {
                if (backup_pvs[i].Count > max_pv_length)
                {
                    max_pv_length = backup_pvs[i].Count;
                    local_pv = backup_pvs[i];
                }
            }

            abt.pv.Clear();
            for (i = 0; i < local_pv.Count; i++)
            {
                abt.pv.Add(local_pv[i]);
            }

            abt.BestValue = best_value;
        }


        public static void AnalyzeWithAlphaBeta(int num_threads, int iteration_max)
        {
            InitKKPIndex();
            Init();
            //Feature2.Init();
            SetThreadNum(thread_num);
            InitTlpAbt(thread_num);
            InitLimitTable();
            Load();

            int color, prev_value;
            List<Record> records = IO.ReadRecordFile("20220403_nhk_hai.txt");
            int limit = records[0].str_moves.Length;
            AlphaBetaTree abt = new AlphaBetaTree();
            Board.BoardTreeAlloc(ref abt.bt);
            Board.Init(ref abt.bt);
            color = 0;

            color = 0;
            for (int i = 0; i < limit; i++)
            {
                Move move = CSA.CSA2Move(abt.bt, records[0].str_moves[i]);

                int sq_king = abt.bt.SQ_King[color];
                var bb = IsAttacked(abt.bt, sq_king, color);
                int record_value = 0;

                Do(ref abt.bt, move, color);
                record_value = Eval(abt.bt);
                if (color == 0)
                    record_value = -record_value;
                UnDo(ref abt.bt, move, color);

                List<Move> moves = new List<Move>();

                if (bb == 0)
                {
                    GenDrop(abt.bt, color, ref moves);
                    GenNoCap(abt.bt, color, ref moves);
                    GenCap(abt.bt, color, ref moves);
                }
                else
                {
                    GenEvasion(abt.bt, color, ref moves);
                }

                List<string> str_out = new List<string>();
                int move_count = moves.Count;

                List<Move> regal_moves = new List<Move>();

                for (int j = 0; j < move_count; j++)
                {
                    // 自玉がDiscovered Checkになってしまった場合
                    Do(ref abt.bt, moves[j], color);
                    if (IsAttacked(abt.bt, abt.bt.SQ_King[color], color) > 0)
                    {
                        UnDo(ref abt.bt, moves[j], color);
                        continue;
                    }
                    int ifrom = moves[j].From;
                    int ito = moves[j].To;
                    int icap_pc = (int)moves[i].CapPiece;

                    if (ifrom < Square_NB)
                    {
                        // PIN駒を動かしてしまった場合
                        Direction idirec = Adirec[ifrom, ito];
                        if (IsPinnedOnKing(abt.bt, ifrom, idirec, color) > 0)
                        {
                            UnDo(ref abt.bt, moves[j], color);
                            continue;
                        }
                    }
                    else
                    {
                        // 駒打ちなのに駒取りになってしまった場合 = > 多分発生しないが一応入れておく
                        if (icap_pc != (int)Piece.Empty)
                        {
                            UnDo(ref abt.bt, moves[j], color);
                            continue;
                        }
                    }

                    // 玉を取ってしまった場合
                    if (icap_pc == (int)Piece.King)
                    {
                        UnDo(ref abt.bt, moves[j], color);
                        continue;
                    }

                    regal_moves.Add(moves[j]);

                    UnDo(ref abt.bt, moves[j], color);
                }

                int regal_move_count = regal_moves.Count;
                List<int> scores = new List<int>();

                for (int j = 0; j < regal_move_count; j++)
                {
                    int v0 = Eval(abt.bt);//KKP1564, KPP-10597
                    Do(ref abt.bt, regal_moves[j], color);

                    // 自玉がDiscovered Checkになってしまった場合
                    /*if (IsAttacked(abt.bt, abt.bt.SQ_King[color], color) > 0)
                    {
                        UnDo(ref abt.bt, regal_moves[j], color);
                        continue;
                    }*/
                    Console.WriteLine();
                    int v = Eval(abt.bt);//KKP1542, KPP-9479
                    int v2 = EvalWrapper(abt.bt, color, abt.bt.ply, regal_moves[j], false);
                    if (color == 1)
                    {
                        v2 = -v2;
                        //v2 /= FV_SCALE;
                    }
                    scores.Add(v2);
                    string s = "v1=" + v.ToString() + ", v2=" + v2.ToString() + ", move=" + Move2CSA(regal_moves[j]);
                    str_out.Add(s);
                    Console.WriteLine(s);
                    Sort.MoveAndScore temp_mas = new Sort.MoveAndScore();
                    temp_mas.move = regal_moves[j];
                    temp_mas.score = v;
                    abt.mas_before[abt.bt.ply, j] = temp_mas;
                    UnDo(ref abt.bt, regal_moves[j], color);
                }

                List<int> ranks = new List<int>();
                for (int j = 0; j < regal_move_count; j++)
                {
                    int idx = scores.IndexOf(scores.Min());
                    ranks.Add(idx);
                    scores[idx] = int.MaxValue;
                }

                int value = 0;
                List<Move>[] RootMoves = new List<Move>[num_threads];
                List<int>[] extensions = new List<int>[num_threads];
                int[] root_move_counts = new int[num_threads];
                int child = move_count / num_threads;
                for (int j = 0; j < num_threads; j++)
                {
                    RootMoves[j] = new List<Move>();
                    extensions[j] = new List<int>();
                }
                if (regal_moves.Count == 1)
                {
                    for (int j = 0; j < num_threads; j++)
                    {
                        RootMoves[j].Add(regal_moves[ranks[0]]);
                        extensions[j].Add(Ply_Inc / 2);
                    }
                }
                else if (regal_moves.Count == 2)
                {
                    for (int j = 0; j < num_threads; j++)
                    {
                        RootMoves[j].Add(regal_moves[ranks[0]]);
                        extensions[j].Add(Ply_Inc / 2);
                    }
                    for (int j = 0; j < num_threads; j++)
                    {
                        RootMoves[j].Add(regal_moves[ranks[1]]);
                        extensions[j].Add(Ply_Inc / 4);
                    }
                }
                else
                {
                    for (int j = 0; j < num_threads; j++)
                    {
                        RootMoves[j].Add(regal_moves[ranks[0]]);
                        extensions[j].Add(Ply_Inc / 2);
                    }
                    for (int j = 0; j < num_threads; j++)
                    {
                        RootMoves[j].Add(regal_moves[ranks[1]]);
                        extensions[j].Add(Ply_Inc / 4);
                    }
                    for (int j = 0; j < num_threads; j++)
                    {
                        RootMoves[j].Add(regal_moves[ranks[2]]);
                        extensions[j].Add(Ply_Inc / 8);
                    }
                }
                int thread_index = 0;
                for (int j = 0; j < regal_move_count; j++)
                {
                    RootMoves[thread_index].Add(regal_moves[ranks[j]]);
                    int extension = 0;// reductionはとりあえずなし。
                    if (j == 0)
                    {
                        extension = Ply_Inc / 2;
                    }
                    else if (j == 1)
                    {
                        extension = Ply_Inc / 4;
                    }
                    else if (j == 2)
                    {
                        extension = Ply_Inc / 8;
                    }
                    extensions[thread_index++].Add(extension);
                    if (thread_index >= num_threads)
                    {
                        thread_index = 0;
                    }
                }

                Sort.MoveAndScore[,] mas_before = new Sort.MoveAndScore[num_threads, 700];
                Sort.MoveAndScore[,] mas_after = new Sort.MoveAndScore[num_threads, 700];
                List<int> temp_value = new List<int>();
                for (int j = 0; j < num_threads; j++)
                {
                    for (int k = 0; k < RootMoves[j].Count; k++)
                    {
                        Do(ref abt.bt, RootMoves[j][k], color);
                        int v = EvalWrapper(abt.bt, color, abt.bt.ply, RootMoves[j][k], false);
                        Sort.MoveAndScore temp_mas = new Sort.MoveAndScore();
                        temp_mas.move = RootMoves[j][k];
                        temp_mas.score = v;
                        temp_value.Add(v);
                        mas_before[j, k] = temp_mas;
                        UnDo(ref abt.bt, RootMoves[j][k], color);
                    }
                }

                temp_value.Sort();

                for (int j = 0; j < num_threads; j++)
                {
                    Sort.MergeSort(ref mas_before, ref mas_after, 0, RootMoves[j].Count, j);
                }

                for (int j = 0; j < num_threads; j++)
                {
                    for (int k = 0; k < RootMoves[j].Count; k++)
                    {
                        var m = mas_after[j, k].move;
                        var s = Move2CSA(m);
                        Console.WriteLine("thread=" + j.ToString() + ", move=" + s + ", score=" + mas_after[j, k].score.ToString());
                    }
                }

                //return;

                //Sort.MergeSort(ref mas_before, ref mas_after, 0, moves.Count, 0);

                Task[] tasks = new Task[num_threads - 1];

                AlphaBetaTree[] abt2 = new AlphaBetaTree[num_threads - 1];

                abt2[0] = new AlphaBetaTree();
                abt2[1] = new AlphaBetaTree();
                abt2[2] = new AlphaBetaTree();
                abt2[3] = new AlphaBetaTree();
                abt2[4] = new AlphaBetaTree();
                abt2[5] = new AlphaBetaTree();
                abt2[6] = new AlphaBetaTree();
                abt2[7] = new AlphaBetaTree();
                abt2[8] = new AlphaBetaTree();
                abt2[9] = new AlphaBetaTree();
                abt2[10] = new AlphaBetaTree();
                abt.task_number = 0;
                abt2[0].task_number = 1;
                abt2[1].task_number = 2;
                abt2[2].task_number = 3;
                abt2[3].task_number = 4;
                abt2[4].task_number = 5;
                abt2[5].task_number = 6;
                abt2[6].task_number = 7;
                abt2[7].task_number = 8;
                abt2[8].task_number = 9;
                abt2[9].task_number = 10;
                abt2[10].task_number = 11;

                Board.BoardTreeAlloc(ref abt2[0].bt);
                Board.Init(ref abt2[0].bt);
                abt2[0].bt = Board.DeepCopy(abt.bt, false);
                //abt2[0].bt = SFEN.ToBoard(str_sfens[index]);
                abt2[0].bt.ply = i + 1;// 後で要確認
                prev_value = Eval(abt.bt);
                abt2[0].iteration_max = iteration_max;
                abt2[0].tt = new TT();

                Board.BoardTreeAlloc(ref abt2[1].bt);
                Board.Init(ref abt2[1].bt);
                abt2[1].bt = Board.DeepCopy(abt.bt, false);
                //abt2[0].bt = SFEN.ToBoard(str_sfens[index]);
                abt2[1].bt.ply = i + 1;// 後で要確認
                //prev_value = Eval(abt.bt);
                abt2[1].iteration_max = iteration_max;
                abt2[1].tt = new TT();

                Board.BoardTreeAlloc(ref abt2[2].bt);
                Board.Init(ref abt2[2].bt);
                abt2[2].bt = Board.DeepCopy(abt.bt, false);
                //abt2[0].bt = SFEN.ToBoard(str_sfens[index]);
                abt2[2].bt.ply = i + 1;// 後で要確認
                //prev_value = Eval(abt.bt);
                abt2[2].iteration_max = iteration_max;
                abt2[2].tt = new TT();

                Board.BoardTreeAlloc(ref abt2[3].bt);
                Board.Init(ref abt2[3].bt);
                abt2[3].bt = Board.DeepCopy(abt.bt, false);
                //abt2[0].bt = SFEN.ToBoard(str_sfens[index]);
                abt2[3].bt.ply = i + 1;// 後で要確認
                //prev_value = Eval(abt.bt);
                abt2[3].iteration_max = iteration_max;
                abt2[3].tt = new TT();

                Board.BoardTreeAlloc(ref abt2[4].bt);
                Board.Init(ref abt2[4].bt);
                abt2[4].bt = Board.DeepCopy(abt.bt, false);
                //abt2[0].bt = SFEN.ToBoard(str_sfens[index]);
                abt2[4].bt.ply = i + 1;// 後で要確認
                //prev_value = Eval(abt.bt);
                abt2[4].iteration_max = iteration_max;
                abt2[4].tt = new TT();

                Board.BoardTreeAlloc(ref abt2[5].bt);
                Board.Init(ref abt2[5].bt);
                abt2[5].bt = Board.DeepCopy(abt.bt, false);
                //abt2[0].bt = SFEN.ToBoard(str_sfens[index]);
                abt2[5].bt.ply = i + 1;// 後で要確認
                //prev_value = Eval(abt.bt);
                abt2[5].iteration_max = iteration_max;
                abt2[5].tt = new TT();

                Board.BoardTreeAlloc(ref abt2[6].bt);
                Board.Init(ref abt2[6].bt);
                abt2[6].bt = Board.DeepCopy(abt.bt, false);
                //abt2[0].bt = SFEN.ToBoard(str_sfens[index]);
                abt2[6].bt.ply = i + 1;// 後で要確認
                //prev_value = Eval(abt.bt);
                abt2[6].iteration_max = iteration_max;
                abt2[6].tt = new TT();

                Board.BoardTreeAlloc(ref abt2[7].bt);
                Board.Init(ref abt2[7].bt);
                abt2[7].bt = Board.DeepCopy(abt.bt, false);
                //abt2[0].bt = SFEN.ToBoard(str_sfens[index]);
                abt2[7].bt.ply = i + 1;// 後で要確認
                //prev_value = Eval(abt.bt);
                abt2[7].iteration_max = iteration_max;
                abt2[7].tt = new TT();

                Board.BoardTreeAlloc(ref abt2[8].bt);
                Board.Init(ref abt2[8].bt);
                abt2[8].bt = Board.DeepCopy(abt.bt, false);
                //abt2[0].bt = SFEN.ToBoard(str_sfens[index]);
                abt2[8].bt.ply = i + 1;// 後で要確認
                //prev_value = Eval(abt.bt);
                abt2[8].iteration_max = iteration_max;
                abt2[8].tt = new TT();

                Board.BoardTreeAlloc(ref abt2[9].bt);
                Board.Init(ref abt2[9].bt);
                abt2[9].bt = Board.DeepCopy(abt.bt, false);
                //abt2[0].bt = SFEN.ToBoard(str_sfens[index]);
                abt2[9].bt.ply = i + 1;// 後で要確認
                //prev_value = Eval(abt.bt);
                abt2[9].iteration_max = iteration_max;
                abt2[9].tt = new TT();

                Board.BoardTreeAlloc(ref abt2[10].bt);
                Board.Init(ref abt2[10].bt);
                abt2[10].bt = Board.DeepCopy(abt.bt, false);
                //abt2[0].bt = SFEN.ToBoard(str_sfens[index]);
                abt2[10].bt.ply = i + 1;// 後で要確認
                //prev_value = Eval(abt.bt);
                abt2[10].iteration_max = iteration_max;
                abt2[10].tt = new TT();

                abt.tt = ttt;
                abt2[0].tt = ttt;
                abt2[1].tt = ttt;
                abt2[2].tt = ttt;
                abt2[3].tt = ttt;
                abt2[4].tt = ttt;
                abt2[5].tt = ttt;
                abt2[6].tt = ttt;
                abt2[7].tt = ttt;
                abt2[8].tt = ttt;
                abt2[9].tt = ttt;
                abt2[10].tt = ttt;

                Move[] best_moves = new Move[num_threads];
                int[] best_values = new int[num_threads];
                //RootLoop(1, ref abt2[0], mas_after, RootMoves[1], iteration_max, ref best_moves[1], ref best_values[1], temp_value[0]);

                tasks[0] = Task.Factory.StartNew(() => RootLoop(1, ref abt2[0], mas_after, RootMoves[1], iteration_max, ref best_moves[1], ref best_values[1], temp_value[0], extensions[1]));
                tasks[1] = Task.Factory.StartNew(() => RootLoop(2, ref abt2[1], mas_after, RootMoves[2], iteration_max, ref best_moves[2], ref best_values[2], temp_value[0], extensions[2]));
                tasks[2] = Task.Factory.StartNew(() => RootLoop(3, ref abt2[2], mas_after, RootMoves[3], iteration_max, ref best_moves[3], ref best_values[3], temp_value[0], extensions[3]));
                tasks[3] = Task.Factory.StartNew(() => RootLoop(4, ref abt2[3], mas_after, RootMoves[4], iteration_max, ref best_moves[4], ref best_values[4], temp_value[0], extensions[4]));
                tasks[4] = Task.Factory.StartNew(() => RootLoop(5, ref abt2[4], mas_after, RootMoves[5], iteration_max, ref best_moves[5], ref best_values[5], temp_value[0], extensions[5]));
                tasks[5] = Task.Factory.StartNew(() => RootLoop(6, ref abt2[5], mas_after, RootMoves[6], iteration_max, ref best_moves[6], ref best_values[6], temp_value[0], extensions[6]));
                tasks[6] = Task.Factory.StartNew(() => RootLoop(7, ref abt2[6], mas_after, RootMoves[7], iteration_max, ref best_moves[7], ref best_values[7], temp_value[0], extensions[7]));
                tasks[7] = Task.Factory.StartNew(() => RootLoop(8, ref abt2[7], mas_after, RootMoves[8], iteration_max, ref best_moves[8], ref best_values[8], temp_value[0], extensions[8]));
                tasks[8] = Task.Factory.StartNew(() => RootLoop(9, ref abt2[8], mas_after, RootMoves[9], iteration_max, ref best_moves[9], ref best_values[9], temp_value[0], extensions[9]));
                tasks[9] = Task.Factory.StartNew(() => RootLoop(10, ref abt2[9], mas_after, RootMoves[10], iteration_max, ref best_moves[10], ref best_values[10], temp_value[0], extensions[10]));
                tasks[10] = Task.Factory.StartNew(() => RootLoop(11, ref abt2[10], mas_after, RootMoves[11], iteration_max, ref best_moves[11], ref best_values[11], temp_value[0], extensions[11]));
                //tasks[11] = Task.Factory.StartNew(() => RootLoop(11, ref abt2[10], mas_after, RootMoves[11], iteration_max, ref best_moves[11], ref best_values[11], temp_value[0]));

                RootLoop(0, ref abt, mas_after, RootMoves[0], iteration_max, ref best_moves[0], ref best_values[0], temp_value[0], extensions[0]);

                //Task.WaitAll(tasks[0], tasks[1]);
                Task.WaitAll(tasks);
                int length = abt.pv.Count;
                string str_pv = "";
                //str_pv += CSA.Move2CSA(best_moves[0]) + ",";
                for (int j = 0; j < length; j++)
                {
                    Move m = abt.pv[j];
                    str_pv += CSA.Move2CSA(m);
                    str_pv += ",";
                }

                Console.WriteLine(str_pv);
                Console.WriteLine("num_hash_cut=" + abt.hash_cut.ToString());
                //best_values[0] = -best_values[0];
                Console.WriteLine("best_value=" + best_values[0].ToString());

                Console.WriteLine();

                length = abt2[0].pv.Count;
                str_pv = "";
                //str_pv += CSA.Move2CSA(best_moves[1]) + ",";
                for (int j = 0; j < length; j++)
                {
                    Move m = abt2[0].pv[j];
                    str_pv += CSA.Move2CSA(m);
                    str_pv += ",";
                }

                //abt.bt.EvalArray[i + 2] = value;
                Console.WriteLine(str_pv);
                Console.WriteLine("num_hash_cut=" + abt.hash_cut.ToString());
                //best_values[1] = -best_values[1];
                Console.WriteLine("best_value=" + best_values[1].ToString());

                Console.WriteLine();

                length = abt2[1].pv.Count;
                str_pv = "";
                //str_pv += CSA.Move2CSA(best_moves[2]) + ",";
                for (int j = 0; j < length; j++)
                {
                    Move m = abt2[1].pv[j];
                    str_pv += CSA.Move2CSA(m);
                    str_pv += ",";
                }

                //abt.bt.EvalArray[i + 2] = value;
                Console.WriteLine(str_pv);
                Console.WriteLine("num_hash_cut=" + abt.hash_cut.ToString());
                //best_values[2] = -best_values[2];
                Console.WriteLine("best_value=" + best_values[2].ToString());

                Console.WriteLine();

                length = abt2[2].pv.Count;
                str_pv = "";
                //str_pv += CSA.Move2CSA(best_moves[3]) + ",";
                for (int j = 0; j < length; j++)
                {
                    Move m = abt2[2].pv[j];
                    str_pv += CSA.Move2CSA(m);
                    str_pv += ",";
                }

                //abt.bt.EvalArray[i + 2] = value;
                Console.WriteLine(str_pv);
                Console.WriteLine("num_hash_cut=" + abt.hash_cut.ToString());
                //best_values[3] = -best_values[3];
                Console.WriteLine("best_value=" + best_values[3].ToString());

                Console.WriteLine();

                length = abt2[3].pv.Count;
                str_pv = "";
                //str_pv += CSA.Move2CSA(best_moves[4]) + ",";
                for (int j = 0; j < length; j++)
                {
                    Move m = abt2[3].pv[j];
                    str_pv += CSA.Move2CSA(m);
                    str_pv += ",";
                }

                //abt.bt.EvalArray[i + 2] = value;
                Console.WriteLine(str_pv);
                Console.WriteLine("num_hash_cut=" + abt.hash_cut.ToString());
                //best_values[4] = -best_values[4];
                Console.WriteLine("best_value=" + best_values[4].ToString());

                Console.WriteLine();

                length = abt2[4].pv.Count;
                str_pv = "";
                //str_pv += CSA.Move2CSA(best_moves[5]) + ",";
                for (int j = 0; j < length; j++)
                {
                    Move m = abt2[4].pv[j];
                    str_pv += CSA.Move2CSA(m);
                    str_pv += ",";
                }

                //abt.bt.EvalArray[i + 2] = value;
                Console.WriteLine(str_pv);
                Console.WriteLine("num_hash_cut=" + abt.hash_cut.ToString());
                //best_values[5] = -best_values[5];
                Console.WriteLine("best_value=" + best_values[5].ToString());

                Console.WriteLine();

                length = abt2[5].pv.Count;
                str_pv = "";
                //str_pv += CSA.Move2CSA(best_moves[6]) + ",";
                for (int j = 0; j < length; j++)
                {
                    Move m = abt2[5].pv[j];
                    str_pv += CSA.Move2CSA(m);
                    str_pv += ",";
                }

                //abt.bt.EvalArray[i + 2] = value;
                Console.WriteLine(str_pv);
                Console.WriteLine("num_hash_cut=" + abt.hash_cut.ToString());
                //best_values[6] = -best_values[6];
                Console.WriteLine("best_value=" + best_values[6].ToString());

                length = abt2[6].pv.Count;
                str_pv = "";
                //str_pv += CSA.Move2CSA(best_moves[6]) + ",";
                for (int j = 0; j < length; j++)
                {
                    Move m = abt2[6].pv[j];
                    str_pv += CSA.Move2CSA(m);
                    str_pv += ",";
                }

                //abt.bt.EvalArray[i + 2] = value;
                Console.WriteLine(str_pv);
                Console.WriteLine("num_hash_cut=" + abt.hash_cut.ToString());
                //best_values[7] = -best_values[7];
                Console.WriteLine("best_value=" + best_values[7].ToString());

                length = abt2[7].pv.Count;
                str_pv = "";
                //str_pv += CSA.Move2CSA(best_moves[7]) + ",";
                for (int j = 0; j < length; j++)
                {
                    Move m = abt2[7].pv[j];
                    str_pv += CSA.Move2CSA(m);
                    str_pv += ",";
                }

                //abt.bt.EvalArray[i + 2] = value;
                Console.WriteLine(str_pv);
                Console.WriteLine("num_hash_cut=" + abt.hash_cut.ToString());
                //best_values[8] = -best_values[8];
                Console.WriteLine("best_value=" + best_values[8].ToString());

                length = abt2[8].pv.Count;
                str_pv = "";
                //str_pv += CSA.Move2CSA(best_moves[8]) + ",";
                for (int j = 0; j < length; j++)
                {
                    Move m = abt2[8].pv[j];
                    str_pv += CSA.Move2CSA(m);
                    str_pv += ",";
                }

                //abt.bt.EvalArray[i + 2] = value;
                Console.WriteLine(str_pv);
                Console.WriteLine("num_hash_cut=" + abt.hash_cut.ToString());
                //best_values[9] = -best_values[9];
                Console.WriteLine("best_value=" + best_values[9].ToString());

                length = abt2[9].pv.Count;
                str_pv = "";
                //str_pv += CSA.Move2CSA(best_moves[9]) + ",";
                for (int j = 0; j < length; j++)
                {
                    Move m = abt2[9].pv[j];
                    str_pv += CSA.Move2CSA(m);
                    str_pv += ",";
                }

                //abt.bt.EvalArray[i + 2] = value;
                Console.WriteLine(str_pv);
                Console.WriteLine("num_hash_cut=" + abt.hash_cut.ToString());
                //best_values[10] = -best_values[10];
                Console.WriteLine("best_value=" + best_values[10].ToString());

                length = abt2[10].pv.Count;
                str_pv = "";
                //str_pv += CSA.Move2CSA(best_moves[10]) + ",";
                for (int j = 0; j < length; j++)
                {
                    Move m = abt2[10].pv[j];
                    str_pv += CSA.Move2CSA(m);
                    str_pv += ",";
                }

                Do(ref abt.bt, move, color);
                color ^= 1;

                break;
            }
        }

        private static void PlayOutLoop(ref BoardTree bt, int color, List<Move> moves, int move_count, ref Tree t, int playout_limit)
        {
            List<Move> temp_moves = new List<Move>();
            int temp_ply = bt.ply;
            for (int j = 0; j < move_count; j++)
            {
                Do(ref bt, moves[j], color);
                t.win_count = 0;
                t.lose_count = 0;
                t.draw_count = 0;
                for (int k = 0; k < playout_limit; k++)
                {
                    BoardTree bt_temp = Board.DeepCopy(bt, false);
                    int iret = PlayOutNoSearch(ref bt_temp, color ^ 1, temp_ply + 1, ref temp_moves);
                    if (iret == 0)
                    {
                        t.win_count++;
                    }
                    else if (iret == 1)
                    {
                        t.lose_count++;
                    }
                    else
                    {
                        t.draw_count++;
                    }
                }
                temp_moves.Clear();
                t.list_win.Add(t.win_count);
                t.list_lose.Add(t.lose_count);
                t.list_draw.Add(t.draw_count);
                UnDo(ref bt, moves[j], color);
            }
        }

        public static void AnalyzeWithPlayOutMultiTask2(string str_sfen, int num_threads, int playout_limit)
        {
            int i;
            string str_out = "";
            Init();
            int color;
            BoardTree bt = new BoardTree();
            Board.BoardTreeAlloc(ref bt);
            Board.Init(ref bt);
            color = 0;

            bt = SFEN.ToBoard(str_sfen);

            color = (int)bt.RootColor;

            StreamWriter sw = IO.OpenStreamWriter("log_play_out_multi_task.txt");

            // 手を生成する
            int move_count = 0;
            List<Move> moves = new List<Move>();
            List<Move> temp_moves = new List<Move>();
            if (IsAttacked(bt, bt.SQ_King[color], color) == 0)
            {
                GenDrop(bt, color, ref temp_moves);
                GenNoCap(bt, color, ref temp_moves);
                GenCap(bt, color, ref temp_moves);
            }
            else
            {
                GenEvasion(bt, color, ref temp_moves);
            }

            for (i = 0; i < temp_moves.Count; i++)
            {
                if (IsMoveValid(ref bt, temp_moves[i], color))
                    moves.Add(temp_moves[i]);
            }

            move_count = moves.Count;

            if (move_count <= num_threads)
            {
                num_threads = move_count;
            }

            // Listや構造体を配列にすると、参照渡しになるようで、動作がおかしくなるため、
            // 個別に変数を用意する。
            List<Move> split_moves0 = new List<Move>();
            List<Move> split_moves1 = new List<Move>();
            List<Move> split_moves2 = new List<Move>();
            List<Move> split_moves3 = new List<Move>();
            List<Move> split_moves4 = new List<Move>();
            List<Move> split_moves5 = new List<Move>();
            List<Move> split_moves6 = new List<Move>();
            List<Move> split_moves7 = new List<Move>();
            List<Move> split_moves8 = new List<Move>();
            List<Move> split_moves9 = new List<Move>();
            List<Move> split_moves10 = new List<Move>();
            List<Move> split_moves11 = new List<Move>();
            List<Move> split_moves12 = new List<Move>();
            List<Move> split_moves13 = new List<Move>();
            List<Move> split_moves14 = new List<Move>();
            List<Move> split_moves15 = new List<Move>();

            int task_number = 0;
            for (i = 0; i < move_count; i++)
            {
                switch (task_number)
                {
                    case 0:
                        split_moves0.Add(moves[i]);
                        break;
                    case 1:
                        split_moves1.Add(moves[i]);
                        break;
                    case 2:
                        split_moves2.Add(moves[i]);
                        break;
                    case 3:
                        split_moves3.Add(moves[i]);
                        break;
                    case 4:
                        split_moves4.Add(moves[i]);
                        break;
                    case 5:
                        split_moves5.Add(moves[i]);
                        break;
                    case 6:
                        split_moves6.Add(moves[i]);
                        break;
                    case 7:
                        split_moves7.Add(moves[i]);
                        break;
                    case 8:
                        split_moves8.Add(moves[i]);
                        break;
                    case 9:
                        split_moves9.Add(moves[i]);
                        break;
                    case 10:
                        split_moves10.Add(moves[i]);
                        break;
                    case 11:
                        split_moves11.Add(moves[i]);
                        break;
                    case 12:
                        split_moves12.Add(moves[i]);
                        break;
                    case 13:
                        split_moves13.Add(moves[i]);
                        break;
                    case 14:
                        split_moves14.Add(moves[i]);
                        break;
                    case 15:
                        split_moves15.Add(moves[i]);
                        break;
                }
                ++task_number;
                if (task_number == num_threads)
                {
                    task_number = 0;
                }
            }

            Tree tree0 = new Tree();
            Tree tree1 = new Tree();
            Tree tree2 = new Tree();
            Tree tree3 = new Tree();
            Tree tree4 = new Tree();
            Tree tree5 = new Tree();
            Tree tree6 = new Tree();
            Tree tree7 = new Tree();
            Tree tree8 = new Tree();
            Tree tree9 = new Tree();
            Tree tree10 = new Tree();
            Tree tree11 = new Tree();
            Tree tree12 = new Tree();
            Tree tree13 = new Tree();
            Tree tree14 = new Tree();
            Tree tree15 = new Tree();

            int temp_ply = bt.ply;

            Task[] tasks = new Task[num_threads];

            BoardTree bt0 = Board.DeepCopy(bt, false);
            tasks[0] = Task.Factory.StartNew(() => PlayOutLoop(ref bt0, color, split_moves0, split_moves0.Count, ref tree0, playout_limit));
            if (num_threads == 1)
            {
                Task.WaitAll(tasks[0]);
                goto next;
            }
            BoardTree bt1 = Board.DeepCopy(bt, false);
            tasks[1] = Task.Factory.StartNew(() => PlayOutLoop(ref bt1, color, split_moves1, split_moves1.Count, ref tree1, playout_limit));
            if (num_threads == 2)
            {
                Task.WaitAll(tasks[0], tasks[1]);
                goto next;
            }
            BoardTree bt2 = Board.DeepCopy(bt, false);
            tasks[2] = Task.Factory.StartNew(() => PlayOutLoop(ref bt2, color, split_moves2, split_moves2.Count, ref tree2, playout_limit));
            if (num_threads == 3)
            {
                Task.WaitAll(tasks[0], tasks[1], tasks[2]);
                goto next;
            }
            BoardTree bt3 = Board.DeepCopy(bt, false);
            tasks[3] = Task.Factory.StartNew(() => PlayOutLoop(ref bt3, color, split_moves3, split_moves3.Count, ref tree3, playout_limit));
            if (num_threads == 4)
            {
                Task.WaitAll(tasks[0], tasks[1], tasks[2], tasks[3]);
                goto next;
            }
            BoardTree bt4 = Board.DeepCopy(bt, false);
            tasks[4] = Task.Factory.StartNew(() => PlayOutLoop(ref bt4, color, split_moves4, split_moves4.Count, ref tree4, playout_limit));
            if (num_threads == 5)
            {
                Task.WaitAll(tasks[0], tasks[1], tasks[2], tasks[3], tasks[4]);
                goto next;
            }
            BoardTree bt5 = Board.DeepCopy(bt, false);
            tasks[5] = Task.Factory.StartNew(() => PlayOutLoop(ref bt5, color, split_moves5, split_moves5.Count, ref tree5, playout_limit));
            if (num_threads == 6)
            {
                Task.WaitAll(tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5]);
                goto next;
            }
            BoardTree bt6 = Board.DeepCopy(bt, false);
            tasks[6] = Task.Factory.StartNew(() => PlayOutLoop(ref bt6, color, split_moves6, split_moves6.Count, ref tree6, playout_limit));
            if (num_threads == 7)
            {
                Task.WaitAll(tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[6]);
                goto next;
            }
            BoardTree bt7 = Board.DeepCopy(bt, false);
            tasks[7] = Task.Factory.StartNew(() => PlayOutLoop(ref bt7, color, split_moves7, split_moves7.Count, ref tree7, playout_limit));
            if (num_threads == 8)
            {
                Task.WaitAll(tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[6], tasks[7]);
                goto next;
            }
            BoardTree bt8 = Board.DeepCopy(bt, false);
            tasks[8] = Task.Factory.StartNew(() => PlayOutLoop(ref bt8, color, split_moves8, split_moves8.Count, ref tree8, playout_limit));
            if (num_threads == 9)
            {
                Task.WaitAll(tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[6], tasks[7], tasks[8]);
                goto next;
            }
            BoardTree bt9 = Board.DeepCopy(bt, false);
            tasks[9] = Task.Factory.StartNew(() => PlayOutLoop(ref bt9, color, split_moves9, split_moves9.Count, ref tree9, playout_limit));
            if (num_threads == 10)
            {
                Task.WaitAll(tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[6], tasks[7], tasks[8], tasks[9]);
                goto next;
            }
            BoardTree bt10 = Board.DeepCopy(bt, false);
            tasks[10] = Task.Factory.StartNew(() => PlayOutLoop(ref bt10, color, split_moves10, split_moves10.Count, ref tree10, playout_limit));
            if (num_threads == 11)
            {
                Task.WaitAll(tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[6], tasks[7], tasks[8], tasks[9], tasks[10]);
                goto next;
            }
            BoardTree bt11 = Board.DeepCopy(bt, false);
            tasks[11] = Task.Factory.StartNew(() => PlayOutLoop(ref bt11, color, split_moves11, split_moves11.Count, ref tree11, playout_limit));
            if (num_threads == 12)
            {
                Task.WaitAll(tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[6], tasks[7], tasks[8], tasks[9], tasks[10], tasks[11]);
                goto next;
            }
            BoardTree bt12 = Board.DeepCopy(bt, false);
            tasks[12] = Task.Factory.StartNew(() => PlayOutLoop(ref bt12, color, split_moves12, split_moves12.Count, ref tree12, playout_limit));
            if (num_threads == 13)
            {
                Task.WaitAll(tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[6], tasks[7], tasks[8], tasks[9], tasks[10], tasks[11], tasks[12]);
                goto next;
            }
            BoardTree bt13 = Board.DeepCopy(bt, false);
            tasks[13] = Task.Factory.StartNew(() => PlayOutLoop(ref bt13, color, split_moves13, split_moves13.Count, ref tree13, playout_limit));
            if (num_threads == 14)
            {
                Task.WaitAll(tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[6], tasks[7], tasks[8], tasks[9], tasks[10], tasks[11], tasks[12], tasks[13]);
                goto next;
            }
            BoardTree bt14 = Board.DeepCopy(bt, false);
            tasks[14] = Task.Factory.StartNew(() => PlayOutLoop(ref bt14, color, split_moves14, split_moves14.Count, ref tree14, playout_limit));
            if (num_threads == 15)
            {
                Task.WaitAll(tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[6], tasks[7], tasks[8], tasks[9], tasks[10], tasks[11], tasks[12], tasks[13], tasks[14]);
                goto next;
            }
            BoardTree bt15 = Board.DeepCopy(bt, false);
            tasks[15] = Task.Factory.StartNew(() => PlayOutLoop(ref bt15, color, split_moves15, split_moves15.Count, ref tree15, playout_limit));
            Task.WaitAll(tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[6], tasks[7], tasks[8], tasks[9], tasks[10], tasks[11], tasks[12], tasks[13], tasks[14], tasks[15]);

        next:

            List<Tree> trees = new List<Tree>();
            trees.Add(tree0);
            trees.Add(tree1);
            trees.Add(tree2);
            trees.Add(tree3);
            trees.Add(tree4);
            trees.Add(tree5);
            trees.Add(tree6);
            trees.Add(tree7);
            trees.Add(tree8);
            trees.Add(tree9);
            trees.Add(tree10);
            trees.Add(tree11);

            List<List<Move>> moves_list = new List<List<Move>>();
            moves_list.Add(split_moves0);
            moves_list.Add(split_moves1);
            moves_list.Add(split_moves2);
            moves_list.Add(split_moves3);
            moves_list.Add(split_moves4);
            moves_list.Add(split_moves5);
            moves_list.Add(split_moves6);
            moves_list.Add(split_moves7);
            moves_list.Add(split_moves8);
            moves_list.Add(split_moves9);
            moves_list.Add(split_moves10);
            moves_list.Add(split_moves11);

            i = 0;
            List<float> win_rates = new List<float>();
            List<Move> best_moves_list = new List<Move>();
            Move[] best_moves = new Move[3];
            foreach (var tree in trees)
            {
                List<Move> temp_moves_list = moves_list[i];
                int j = 0;
                foreach (var move in temp_moves_list)
                {
                    string str_move = Move2CSA(move);
                    float win_rate = (float)(tree.list_win[j] + tree.list_draw[j] * 0.5) / (tree.list_win[j] + tree.list_lose[j] + tree.list_draw[j]);
                    win_rates.Add(win_rate);
                    best_moves_list.Add(move);
                    str_out = str_move + ", win_rate=" + win_rate.ToString("F2") +
                        ", win_count=" + tree.list_win[j].ToString() +
                        ", lose_count=" + tree.list_lose[j].ToString() +
                        ", draw_count=" + tree.list_draw[j].ToString();
                    sw.WriteLine(str_out);
                    Console.WriteLine(str_out);
                    j++;
                }
                i++;
            }

            float win_rate_1st = new float();
            float win_rate_2nd = new float();
            float win_rate_3rd = new float();

            win_rate_1st = win_rates.Max();
            int index_1st = win_rates.IndexOf(win_rate_1st);
            best_moves[0] = best_moves_list[index_1st];
            win_rates[index_1st] = -1.0f; // 1位を除外
            win_rate_2nd = win_rates.Max();
            int index_2nd = win_rates.IndexOf(win_rate_2nd);
            best_moves[1] = best_moves_list[index_2nd];
            win_rates[index_2nd] = -1.0f; // 2位を除外
            win_rate_3rd = win_rates.Max();
            int index_3rd = win_rates.IndexOf(win_rate_3rd);
            best_moves[2] = best_moves_list[index_3rd];

            Console.WriteLine();

            str_out = "";
            str_out += "best_move_1st=" + Move2CSA(best_moves[0]) + ", win_rate=" + win_rate_1st.ToString("F2") + ", ";
            str_out += "best_move_2nd=" + Move2CSA(best_moves[1]) + ", win_rate=" + win_rate_2nd.ToString("F2") + ", ";
            str_out += "best_move_3rd=" + Move2CSA(best_moves[2]) + ", win_rate=" + win_rate_3rd.ToString("F2");
            sw.WriteLine(str_out);
            Console.WriteLine(str_out);

            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            sw.Close();
        }
    }
}
