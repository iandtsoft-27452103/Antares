using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Antares.Common;

namespace Antares
{
    public class Move
    {
        private uint _move;
        public Move()
        {
            _move = 0U;
        }
        public uint Value
        {
            set { _move = value; }
            get { return _move; }
        }

        public int To
        {
            get
            {
                return (int)_move & 0x007f;
            }
        }

        public int From
        {
            get
            {
                return (int)(_move >> 7) & 0x007f;
            }
        }

        public int FlagPromo
        {
            get
            {
                return (int)(_move >> 14) & 1;
            }
        }

        public Piece PieceType
        {
            get
            {
                return (Piece)((_move >> 15) & 0x000f);
            }
        }

        public Piece CapPiece
        {
            get
            {
                return (Piece)((_move >> 19) & 0x000f);
            }
        }

        public void Clear()
        {
            _move = 0U;
        }

        /*
        xxxxxxxx xxxxxxxx x1111111 To位置
        xxxxxxxx xx111111 1xxxxxxx From位置
        xxxxxxxx x1xxxxxx xxxxxxxx 成る手かどうか
        xxxxx111 1xxxxxxx xxxxxxxx 動かした駒の種類
        x1111xxx xxxxxxxx xxxxxxxx 捕獲した駒
        */

        public void Pack(int from, int to, Piece pc, Piece cap_pc, int flag_promo)
        {
            _move = ((uint)cap_pc << 19) | ((uint)pc << 15) | ((uint)flag_promo << 14) | ((uint)from << 7) | (uint)to;
        }

        public void SetNullMove()
        {
            _move = (1 << 23);
        }
    }
}
