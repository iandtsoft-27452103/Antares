using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Rand = System.UInt128;
using static Antares.Common;

namespace Antares
{
    public class TT
    {
        public Dictionary<Rand, int> value = new Dictionary<Rand, int>();
        public Dictionary<Rand, Common.Color> color = new Dictionary<Rand, Common.Color>();
        public Dictionary<Rand, bool> is_check = new Dictionary<Rand, bool>();
        public Dictionary<Rand, Move> move = new Dictionary<Rand, Move>();// 要不要を後で精査する。
        public Dictionary<Rand, int> ply = new Dictionary<Rand, int>();

        public void Clear()
        {
            value.Clear();
            color.Clear();
            is_check.Clear();
            move.Clear();
            ply.Clear();
        }

        public void Store(Rand k, int v, Common.Color c, bool ch, Move m, int param_ply)
        {
            try
            {
                lock(value){
                    value[k] = v;
                }
                lock (color)
                {
                    color[k] = c;
                }
                lock(is_check)
                {
                    is_check[k] = ch;
                }
                lock(move)
                {
                    move[k] = m;
                }
                lock(ply)
                {
                    ply[k] = param_ply;
                }
            }
            catch (Exception ex)
            {
                throw;
            }
        }
    }
}
