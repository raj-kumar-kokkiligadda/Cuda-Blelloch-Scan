

// 1 << (lg(x - 1) + 1) 
unsigned int NextPow2(const unsigned int& x)
{
    unsigned int y = x;
    y--;
    y |= y >> 1;
    y |= y >> 2;
    y |= y >> 4;
    y |= y >> 8;
    y |= y >> 16;
    y++;
    return y;
}

unsigned char log2(const unsigned int& x)
{
    if(x == 0)
        return 0;
    unsigned int y = x;
    unsigned char count = 0;
    while(!(y & 0x01))
    {
        y = y >> 1;
        count++;
    }
    return count;
}