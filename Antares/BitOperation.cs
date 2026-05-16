using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BitBoard = System.UInt128;
using static Antares.Common;

namespace Antares
{
    public static class BitOperation
    {
        public static int Square(BitBoard bb)
        {
            BitBoard y = BitBoard.TrailingZeroCount(bb);
            return Square_NB - 1 - ((int)y);
        }
    }
}
